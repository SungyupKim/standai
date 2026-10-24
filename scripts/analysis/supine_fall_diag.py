"""Why do supine starts that REACH the hand-over region not end up standing? Runs the whole system (GetUpController, supine side, no hand-over policy) on fixed seeds and
records, at the moment control first leaves the floor policy (the hand-over), the body state (torso/head height, pitch, pitch rate, COM ahead of the front foot, COM forward speed,
feet on the ground) and the outcome (standing at 20 s / stood once / never stood), plus the direction of the first stand-policy collapse. Prints per-feature means by outcome and AUC.
Usage: .venv/bin/python -I scripts/analysis/supine_fall_diag.py [--n 400] [--seed0 58000] [--workers 8] [--side supine]"""
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts"); sys.path.insert(0, ROOT + "scripts/analysis")
import numpy as np

FEATS = ["t_ho", "torso", "head", "pitch", "pitch_rate", "com_ahead", "com_vx", "feet_down", "torso_vx"]


def feats(env, ctl, prev, dt=0.02):
    d = env.data
    m = env._a_masses
    com = float(np.sum(m * d.xipos[env._a_bodies, 0]) / m.sum())
    feet = [d.site_xpos[s] for s in env.a_feet]
    pitch = float(np.degrees(ctl.pitch(env)))
    cur = dict(com=com, pitch=pitch, torso_x=float(d.xpos[env.a_torso_id][0]))
    vx = (com - prev["com"]) / dt if prev else 0.0
    pr = (pitch - prev["pitch"]) / dt if prev else 0.0
    tvx = (cur["torso_x"] - prev["torso_x"]) / dt if prev else 0.0
    return cur, [float(d.xpos[env.a_torso_id][2]), float(d.xpos[env.a_head_id][2]), pitch, pr, com - max(f[0] for f in feet), vx, float(sum(f[2] < 0.04 for f in feet)), tvx]


def run_chunk(args):
    side, seeds, stand, supine, prone = args
    from controller import GetUpController
    from floor_env import FloorEnv
    env = FloorEnv(standready=True, level=0.7, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False,
                   prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    ctl = GetUpController(supine, prone, stand, fall_back=True)
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed); ctl.reset(env)
        prev, ho = None, None; hist = []; fall = None; falls0 = 0
        for t in range(1000):
            obs, _, term, trunc, info = env.step(ctl.act(obs, env)); ctl.observe(env, info)
            cur, f = feats(env, ctl, prev, 0.02)
            if ho is None and ctl.mode != "floor":
                ho = [t * 0.02] + f
            if ho is not None and fall is None and ctl.falls > falls0:
                fall = [t * 0.02 - ho[0]] + f      # state at the first collapse (feature order as above)
            hist.append(float(env.data.xpos[env.a_torso_id][2]) >= 0.95 and abs(cur["pitch"]) < 30)
            prev = cur
        h = np.array(hist); run = np.convolve(h, np.ones(25), mode="valid") >= 25
        rows.append((seed, ho, fall, bool(run.any()), bool(run[-1])))
    return rows


def auc(x, y):      # P(x_pos > x_neg), rank based
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.all() or (~y).all(): return float("nan")
    r = np.argsort(np.argsort(x)) + 1.0
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--seed0", type=int, default=58000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--side", default="supine")
    ap.add_argument("--stand", default=ROOT + "checkpoints/best_stand_big1.zip")
    ap.add_argument("--supine", default=ROOT + "checkpoints/best_floor_supine_fdist.zip"); ap.add_argument("--prone", default=ROOT + "checkpoints/best_prone8.zip")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.n)); ch = [seeds[i::a.workers] for i in range(a.workers)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for part in ex.map(run_chunk, [(a.side, c, a.stand, a.supine, a.prone) for c in ch]) for r in part]
    reached = [r for r in rows if r[1] is not None]
    print(f"{a.side}: {len(rows)} starts, reached the hand-over {len(reached)}, standing at 20 s {sum(r[4] for r in rows)}, of the reached {sum(r[4] for r in reached)}")
    up = np.array([r[4] for r in reached]); X = np.array([r[1] for r in reached])
    print(f"\nstate at the hand-over (n={len(reached)}: {up.sum()} end standing / {(~up).sum()} do not)\n{'feature':12s} {'standing':>10s} {'not':>10s} {'AUC(stand)':>11s}")
    for i, nme in enumerate(FEATS):
        print(f"{nme:12s} {X[up, i].mean():10.3f} {X[~up, i].mean():10.3f} {auc(X[:, i], up):11.2f}")
    fl = [r for r in reached if r[2] is not None]
    print(f"\nfirst stand-policy collapse: {len(fl)} of {len(reached)} reached starts; of those end standing {sum(r[4] for r in fl)}")
    if fl:
        F = np.array([r[2] for r in fl])
        print(f"  time after the hand-over: median {np.median(F[:, 0]):.2f} s (quartiles {np.percentile(F[:, 0], 25):.2f}/{np.percentile(F[:, 0], 75):.2f})")
        print(f"  at the collapse pitch>0 (leaning one way) {np.mean(F[:, 4] > 0):.0%}, COM ahead of the front foot {np.mean(F[:, 6] > 0):.0%}, COM forward speed > 0 {np.mean(F[:, 7] > 0):.0%}")
    np.save(ROOT + f"checkpoints/analysis/fall_diag_{a.side}.npy", np.array([[r[0], *(r[1] or [np.nan] * 9), r[3], r[4]] for r in rows], dtype=float))


if __name__ == "__main__":
    main()
