"""Is the hand-over success condition (handover_env.ok_state) a state the stand policy catches?
Over a sample of pool states: share satisfying the condition AT THE START, and the stand policy's catch (benchmark criterion) from those vs the rest.
Usage: STANDAI_NO_RECOVERY_GATE=1 .venv/bin/python -I scripts/analysis/handover_cond_check.py [--stand CKPT] [--n 3000]"""
import argparse, sys
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
import numpy as np, mujoco
from stable_baselines3 import PPO
from handover_env import HandoverEnv

ap = argparse.ArgumentParser()
ap.add_argument("--stand", default=ROOT + "checkpoints/best_stand_v55b.zip")
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--pool", default=ROOT + "checkpoints/pool_big_all.npz")
a = ap.parse_args()
d = np.load(a.pool, allow_pickle=True)
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(d["qpos"]), a.n, replace=False))
model = PPO.load(a.stand, device="cpu")
env = HandoverEnv()
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
rows = []
for i in idx:
    env.reset(seed=int(i))
    env.data.qpos[:], env.data.qvel[:] = d["qpos"][i], d["qvel"][i]
    mujoco.mj_forward(env.model, env.data); env._prev_com_x = None
    ok0 = env.ok_state()
    obs = env._obs(); run = 0
    for _ in range(150):
        act, _ = model.predict(obs, deterministic=True)
        obs, *_ = env.step(act)
        good = (env.data.xpos[env.a_torso_id][2] >= 0.7 and env.data.xpos[env.a_head_id][2] >= 0.9
                and abs(wrap(env.data.qpos[env.a_ry_qpos])) < np.radians(40))
        run = run + 1 if good else 0
    rows.append((i >= 16000, ok0, run >= 20))
R = np.array(rows)
for name, m in (("supine", ~R[:, 0]), ("prone", R[:, 0])):
    s = R[m]; ok = s[:, 1]
    print(f"{name}: n={len(s)}  condition at start {ok.mean():.1%}  | stand catch: cond {s[ok,2].mean():.0%} (n={ok.sum()})  not-cond {s[~ok,2].mean():.0%} (n={(~ok).sum()})")
