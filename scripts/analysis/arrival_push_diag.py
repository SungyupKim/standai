"""Is the forward COM speed at the hand-over CAUSAL for the stand policy's catch? Collects supine arrival states of the supine floor policy (deterministic, level 0.7), then runs the stand policy from each of them
with the whole body's x velocity changed by delta (m/s; root_x dof, i.e. the COM speed changes by exactly delta, nothing else) and counts the catches (benchmark criterion: torso >= 0.7 m, head >= 0.9 m, |pitch| < 40 deg
for the last 20 of 150 steps). Control: delta = +-0.02 m/s (random sign) shows how often a negligible nudge already flips the outcome (chaos). Reports the catch rate per delta, how many baseline failures are rescued and
baseline catches lost, split by the arrival's own COM speed. Usage: .venv/bin/python -I scripts/analysis/arrival_push_diag.py [--n 400] [--seed0 400000] [--floor CKPT] [--stand CKPT] [--workers 8]"""
import argparse, os, sys
from multiprocessing import Pool
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
import numpy as np

DELTAS = [("ctl+", 0.02), ("ctl-", -0.02), ("-0.4", -0.4), ("-0.2", -0.2), ("0", 0.0), ("+0.1", 0.1), ("+0.2", 0.2), ("+0.3", 0.3), ("+0.5", 0.5), ("+0.8", 0.8)]
_G = {}


def collect(args):
    floor, seeds = args
    from stable_baselines3 import PPO
    from floor_env import FloorEnv
    m = PPO.load(floor, device="cpu")
    env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=450, down_timeout=False, level=0.7, speed_pen=0.08, speed_vz_w=0.25, supine_frac=1.0)
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        if float(env.data.xpos[env.a_torso_id][2]) >= 0.5:
            continue
        for t in range(450):
            a, _ = m.predict(obs, deterministic=True); obs, _, te, tr, info = env.step(a)
            if info.get("floor_success"):
                q, v, _h = info["handoff_state"]
                out.append((s, q.copy(), v.copy())); break
            if te or tr:
                break
    return out


def catch(args):
    stand, q, v, delta, i = args
    import mujoco
    from stable_baselines3 import PPO
    from locomotion_env import LocomotionEnv
    if "m" not in _G:
        _G["m"] = PPO.load(stand, device="cpu"); _G["env"] = LocomotionEnv(stand_only=True); _G["env"].set_calm_frac(1.0)
    env, m = _G["env"], _G["m"]
    env.reset(seed=i)
    env.data.qpos[:], env.data.qvel[:] = q, v
    env.data.qvel[env._a_root_x_dof] += delta
    mujoco.mj_forward(env.model, env.data); env._prev_com_x = None
    obs = env._obs()
    com_vx0 = None
    run = 0
    for _ in range(150):
        act, _ = m.predict(obs, deterministic=True)
        obs, *_ = env.step(act)
        z = float(env.data.xpos[env.a_torso_id][2])
        pitch = (env.data.qpos[env.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi
        ok = z >= 0.7 and float(env.data.xpos[env.a_head_id][2]) >= 0.9 and abs(float(pitch)) < np.deg2rad(40)
        run = run + 1 if ok else 0
    return run >= 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--seed0", type=int, default=400000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--floor", default=ROOT + "checkpoints/best_floor_supine_fdist.zip"); ap.add_argument("--stand", default=ROOT + "checkpoints/best_stand_big1.zip")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.n)); ch = [seeds[i::a.workers] for i in range(a.workers)]
    with Pool(a.workers) as p:
        states = sorted([x for part in p.map(collect, [(a.floor, c) for c in ch]) for x in part], key=lambda t: t[0])
        print(f"{len(states)} arrivals from {a.n} supine starts ({len(states) / a.n:.0%})", flush=True)
        rng = np.random.default_rng(0)
        res = {}
        for name, d in DELTAS:
            jobs = [(a.stand, q, v, d if name not in ("ctl+", "ctl-") else d * (1 if rng.random() < 0.5 else -1), i) for i, (s, q, v) in enumerate(states)]
            res[name] = np.array(p.map(catch, jobs), dtype=bool)
    base = res["0"]
    com_vx = np.array([0.0] * len(states))
    print(f"\nbaseline (delta 0): stand policy catches {base.mean():.1%} ({base.sum()}/{len(base)})")
    print(f"\n{'delta m/s':>10} {'catch':>7} {'rescued':>14} {'lost':>14} {'flip total':>11}")
    for name, d in DELTAS:
        r = res[name]
        resc = (~base & r).sum(); lost = (base & ~r).sum()
        print(f"{name:>10} {r.mean():7.1%} {resc:>4d}/{(~base).sum():<4d} {resc / max((~base).sum(), 1):5.0%} {lost:>4d}/{base.sum():<4d} {lost / max(base.sum(), 1):5.0%} {(r != base).mean():10.1%}")
    np.save(ROOT + "checkpoints/analysis/arrival_push.npy", np.array([res[n] for n, _ in DELTAS]))


if __name__ == "__main__":
    main()
