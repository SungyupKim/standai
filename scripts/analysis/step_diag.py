"""Does the stand policy ever STEP on hand-off states it fails to catch?  (stochastic rollouts, same noise as in training)

For each prone (or supine) state of an eval set, `--reps` stochastic 3 s rollouts. Per rollout: caught (benchmark criterion), and
  overtake = max_t (min foot x) - (max foot x at start)   > 0.02 m: the trailing foot got in front of where the front foot started = a real step
  advance  = max_t (max foot x) - (max foot x at start)   front foot travel
Reported by COM-ahead bin (COM x - front foot x at the start) and caught / missed.
Usage: STANDAI_NO_RECOVERY_GATE=1 .venv/bin/python -I scripts/analysis/step_diag.py [--stand CKPT] [--side prone|supine] [--reps 5]
"""
import argparse, os, sys
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
import numpy as np, mujoco
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv

ap = argparse.ArgumentParser()
ap.add_argument("--stand", default=ROOT + "checkpoints/best_stand_v54.zip")
ap.add_argument("--side", default="prone")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--eval", default=ROOT + "checkpoints/eval2_all.npz")
a = ap.parse_args()
d = np.load(a.eval, allow_pickle=True)
sl = slice(0, 150) if a.side == "supine" else slice(150, 300)
Q, V = d["qpos"][sl], d["qvel"][sl]
model = PPO.load(a.stand, device="cpu")
env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0)
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
fx = lambda: [env.data.site_xpos[s][0] for s in env.a_feet]
rows = []
for i in range(len(Q)):
    for r in range(a.reps):
        env.reset(seed=1000 * i + r)
        env.data.qpos[:], env.data.qvel[:] = Q[i], V[i]
        mujoco.mj_forward(env.model, env.data)
        env._prev_com_x = None
        obs = env._obs()
        f0 = max(fx())
        com = np.sum(env._a_masses * env.data.xipos[env._a_bodies, 0]) / env._a_masses.sum()
        ahead0 = com - f0
        run = 0; adv = -9; ov = -9
        for t in range(150):
            act, _ = model.predict(obs, deterministic=False)
            obs, *_ = env.step(act)
            f = fx(); adv = max(adv, max(f) - f0); ov = max(ov, min(f) - f0)
            ok = (env.data.xpos[env.a_torso_id][2] >= 0.7 and env.data.xpos[env.a_head_id][2] >= 0.9
                  and abs(wrap(env.data.qpos[env.a_ry_qpos])) < np.radians(40))
            run = run + 1 if ok else 0
        rows.append((ahead0, run >= 20, adv, ov))
R = np.array(rows, dtype=float)
print(f"{a.side}: {len(Q)} states x {a.reps} reps, caught {R[:,1].mean():.0%}")
bins = [(-9, -0.05), (-0.05, 0), (0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 9)]
print("COM-ahead bin      n  caught | missed: median advance, step(overtake>2cm) | caught: median advance, step")
for lo, hi in bins:
    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
    if not m.any(): continue
    c, x = m & (R[:, 1] == 1), m & (R[:, 1] == 0)
    f = lambda k: (f"{np.median(R[k,2]):+.2f} m, {np.mean(R[k,3] > 0.02):4.0%}" if k.any() else "   -   ")
    print(f"{lo:+.2f}..{hi:+.2f} {m.sum():5d}  {c.sum()/m.sum():5.0%} | {f(x)} | {f(c)}")
