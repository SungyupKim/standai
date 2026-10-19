"""Hand-over policy -> stand policy chain on hand-off eval states (deterministic). Per COM-ahead bin: hand-over success, stand catch after it, and the end-to-end catch
(benchmark criterion, 150 steps after the switch) next to the stand policy alone.
Usage: STANDAI_NO_RECOVERY_GATE=1 .venv/bin/python -I scripts/analysis/handover_eval.py HANDOVER_CKPT [--stand CKPT] [--set eval2_all] [--side prone|supine]"""
import argparse, sys
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
import numpy as np, mujoco
from stable_baselines3 import PPO
from handover_env import HandoverEnv

ap = argparse.ArgumentParser()
ap.add_argument("handover")
ap.add_argument("--stand", default=ROOT + "checkpoints/best_stand_v55b.zip")
ap.add_argument("--set", default="eval2_all")
ap.add_argument("--side", default="prone")
a = ap.parse_args()
d = np.load(ROOT + f"checkpoints/{a.set}.npz", allow_pickle=True)
n = len(d["qpos"]); sl = range(n // 2, n) if a.side == "prone" else range(0, n // 2)
ho, st = PPO.load(a.handover, device="cpu"), PPO.load(a.stand, device="cpu")
env = HandoverEnv()
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi

def catch(env, obs):
    run = 0
    for _ in range(150):
        act, _ = st.predict(obs, deterministic=True)
        obs, *_ = env.step(act)
        good = (env.data.xpos[env.a_torso_id][2] >= 0.7 and env.data.xpos[env.a_head_id][2] >= 0.9
                and abs(wrap(env.data.qpos[env.a_ry_qpos])) < np.radians(40))
        run = run + 1 if good else 0
    return run >= 20

rows = []
for i in sl:
    env.reset(seed=i); env.data.qpos[:], env.data.qvel[:] = d["qpos"][i], d["qvel"][i]
    mujoco.mj_forward(env.model, env.data); env._prev_com_x = None; env._ho_hold = 0; env._ho_prev_phi = None
    m = env.com_state(); ahead = m[0] - m[1]
    obs = env._obs(); alone = None
    # stand policy alone from the same state
    q0, v0 = env.data.qpos.copy(), env.data.qvel.copy()
    alone = catch(env, obs)
    env.reset(seed=i); env.data.qpos[:], env.data.qvel[:] = q0, v0
    mujoco.mj_forward(env.model, env.data); env._prev_com_x = None; env._ho_hold = 0; env._ho_prev_phi = None
    obs = env._obs(); succ = False
    for t in range(300):
        act, _ = ho.predict(obs, deterministic=True)
        obs, r, te, tr, info = env.step(act)
        if te or tr:
            succ = bool(info.get("ho_success")); break
    chain = catch(env, obs) if succ else False
    rows.append((ahead, alone, succ, chain))
R = np.array(rows, dtype=float)
print(f"{a.side} ({len(R)} states, {a.set}): stand alone {R[:,1].mean():.0%} | hand-over success {R[:,2].mean():.0%}, stand catch after success {R[R[:,2]==1,3].mean():.0%} | END-TO-END chain {R[:,3].mean():.0%}")
for lo, hi in [(-9, -0.05), (-0.05, 0), (0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 9)]:
    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
    if m.any(): print(f"  COM-ahead {lo:+.2f}..{hi:+.2f} n={m.sum():3d}: alone {R[m,1].mean():4.0%}  hand-over ok {R[m,2].mean():4.0%}  chain {R[m,3].mean():4.0%}")
