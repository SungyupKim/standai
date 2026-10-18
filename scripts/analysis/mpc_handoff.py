"""Can this body hold a stand from the hand-off states the stand policy FAILS to catch?  (physics oracle, not a teacher)

1. The stand policy (default v37@96M) is run closed-loop from every state of the held-out strict hand-off set
   (checkpoints/eval_strict_all.npz) with the benchmark's criterion (stand_progress_notify.handoff_catch: after 150 steps torso >= 0.7 m,
   head >= 0.9 m, |pitch| < 40 deg for the last 20 steps).
2. Receding-horizon predictive sampling (the env's own physics; same idea as getup_mpc.py) is run from the policy's FAILED states
   (and from a few policy successes as a calibration: it should catch those too) with the same criterion on the open-loop replay of the
   chosen actions through the real env.step.
   MPC catches most failures  -> the body can do it, the policy cannot (learning problem: guidance / imitation / exploration).
   MPC catches few            -> those states are physically hard; fix it upstream (floor policy hands over better states).

Usage: .venv/bin/python -I scripts/analysis/mpc_handoff.py [--stand CKPT] [--n-fail 60] [--n-ok 20] [--workers 12] [--horizon 25] [--samples 96] [--tag NAME]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
sys.path.insert(0, ROOT + "scripts/analysis")
import numpy as np
import mujoco

EVAL = ROOT + "checkpoints/eval_strict_all.npz"
STAND = ROOT + "checkpoints/autosave/ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip"
WIN = 150
ND = 12
KN = 6
SIGMA = 0.45
PITCH_OK = np.radians(40.0)

_G = {}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def init_worker(horizon, samples):
    import torch
    torch.set_num_threads(1)
    import getup_cem
    getup_cem.init_worker()
    from locomotion_env import LocomotionEnv
    env = LocomotionEnv(stand_only=True)
    env.set_calm_frac(1.0)
    d = np.load(os.environ.get("MPC_SOURCE", EVAL), allow_pickle=True)
    _G.update(env=env, qpos=d["qpos"], qvel=d["qvel"], H=horizon, N=samples)


def load_state(env, i):
    env.reset(seed=i)
    env.data.qpos[:], env.data.qvel[:] = _G["qpos"][i], _G["qvel"][i]
    mujoco.mj_forward(env.model, env.data)
    env._prev_com_x = None


def good(env):
    z = float(env.data.xpos[env.a_torso_id][2])
    h = float(env.data.xpos[env.a_head_id][2])
    p = abs(wrap(env.data.qpos[env.a_ry_qpos]))
    return z >= 0.7 and h >= 0.9 and p < PITCH_OK, z


def policy_catch(args):
    """closed-loop policy from state i; returns True if the benchmark criterion holds."""
    i, ckpt = args
    from stable_baselines3 import PPO
    if "model" not in _G:
        _G["model"] = PPO.load(ckpt, device="cpu")
    env, model = _G["env"], _G["model"]
    load_state(env, i)
    obs = env._obs()
    run = 0
    for _ in range(WIN):
        a, _ = model.predict(obs, deterministic=True)
        obs, *_ = env.step(a)
        run = run + 1 if good(env)[0] else 0
    return i, run >= 20


def apply_ctrl(env, a):
    from env import FRAME_SKIP
    import getup_cem as g
    d = env.data
    for _ in range(FRAME_SKIP):
        d.ctrl[env.a_act] = a
        d.ctrl[env.a_act[env.a_arm_act_idx]] *= env._power_scale(env.a_arm_qpos, env.a_arm_range, g.C['ARM'])
        d.ctrl[env.a_act[env.a_leg_act_idx]] *= env._power_scale(env.a_leg_qpos, env.a_leg_range, g.C['LEG'])
        d.ctrl[env.a_root_act] = 0.0
        d.ctrl[env.a_balance_act] = 0.0
        mujoco.mj_step(env.model, d)


def trace(env):
    d = env.data
    return (d.xpos[env.a_torso_id][2], d.qpos[env.a_ry_qpos], d.xpos[env.a_head_id][2])


def cost(tr):
    z, ry, hz = tr
    ok = (z >= 0.7) & (hz >= 0.9) & (np.abs(wrap(ry)) < PITCH_OK)
    return ok.mean() + 2.0 * ok[-1] + 0.5 * (np.minimum(z, 1.1) / 1.1).mean() - 2.0 * (z.min() < 0.65)


def mpc_state(i):
    env, H, N = _G["env"], _G["H"], _G["N"]
    m = env.model
    load_state(env, i)
    cur = mujoco.MjData(m)
    mujoco.mj_copyData(cur, m, env.data)
    rng = np.random.default_rng(i)
    plan = np.zeros((H, ND))
    xs_k, tt = np.linspace(0, H - 1, KN), np.arange(H)
    acts = []
    for _ in range(WIN):
        noise = rng.standard_normal((N, KN, ND)) * SIGMA
        noise[0] = 0.0
        cands = np.empty((N, H, ND))
        for j in range(ND):
            for k in range(N):
                cands[k, :, j] = plan[:, j] + np.interp(tt, xs_k, noise[k, :, j])
        cands = np.clip(cands, -1, 1)
        best, best_c = 0, -1e9
        for k in range(N):
            mujoco.mj_copyData(env.data, m, cur)
            tr = np.empty((3, H))
            for t in range(H):
                apply_ctrl(env, cands[k, t])
                tr[:, t] = trace(env)
            c = cost(tr)
            if c > best_c:
                best_c, best = c, k
        plan = cands[best]
        a0 = plan[0].copy()
        mujoco.mj_copyData(env.data, m, cur)
        apply_ctrl(env, a0)
        mujoco.mj_copyData(cur, m, env.data)
        acts.append(a0)
        plan = np.vstack([plan[1:], plan[-1:]])
    # verify: open-loop replay of the logged actions through the real env.step, with the benchmark criterion
    load_state(env, i)
    run = 0
    zmin = 9.0
    obs = env._obs()
    obs_log = []
    for a in acts:
        obs_log.append(obs)
        obs, *_ = env.step(np.concatenate([[0.0], a]).astype(np.float32))
        ok, z = good(env)
        run = run + 1 if ok else 0
        zmin = min(zmin, z)
    out = dict(i=i, caught=bool(run >= 20), zmin=float(zmin), z_end=float(z))
    if os.environ.get("MPC_SAVE_DEMOS") == "1":
        out["obs"] = np.array(obs_log, dtype=np.float32)
        out["acts"] = np.array(acts, dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stand", default=STAND)
    ap.add_argument("--n-fail", type=int, default=60)
    ap.add_argument("--n-ok", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--tag", default="v37")
    ap.add_argument("--only", type=int, nargs="*", help="skip the policy pass, run MPC on these state indices")
    a = ap.parse_args()
    t0 = time.time()
    n = len(np.load(EVAL, allow_pickle=True)["qpos"])
    with Pool(a.workers, initializer=init_worker, initargs=(a.horizon, a.samples)) as p:
        if a.only:
            fails, oks = a.only, []
        else:
            res = dict(p.map(policy_catch, [(i, a.stand) for i in range(n)]))
            fails = [i for i in range(n) if not res[i]]
            oks = [i for i in range(n) if res[i]]
            print(f"policy {os.path.basename(a.stand)}: caught {len(oks)}/{n}, failed {len(fails)}  [{time.time() - t0:.0f}s]", flush=True)
            rng = np.random.default_rng(0)
            fails = sorted(rng.choice(fails, min(a.n_fail, len(fails)), replace=False).tolist())
            oks = sorted(rng.choice(oks, min(a.n_ok, len(oks)), replace=False).tolist())
        jobs = [(i, "fail") for i in fails] + [(i, "ok") for i in oks]
        out = []
        for r in p.imap_unordered(mpc_state, [i for i, _ in jobs]):
            kind = dict(jobs)[r["i"]]
            r["policy"] = kind
            out.append(r)
            print(f"state {r['i']:>3} ({kind:>4}): MPC caught {r['caught']}, min torso {r['zmin']:.2f}, end {r['z_end']:.2f}  [{time.time() - t0:.0f}s]", flush=True)
    for kind in ("fail", "ok"):
        rows = [r for r in out if r["policy"] == kind]
        if rows:
            c = sum(r["caught"] for r in rows)
            print(f"MPC on policy-{kind} states: caught {c}/{len(rows)} ({c / len(rows):.0%})")
    os.makedirs(ROOT + "checkpoints/analysis", exist_ok=True)
    json.dump(out, open(ROOT + f"checkpoints/analysis/mpc_handoff_{a.tag}.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
