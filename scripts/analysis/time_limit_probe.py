"""Lying-only reach vs episode length: first step at which the stand-ready region is reached (deterministic, fixed seeds),
and whether the stand policy then holds 3 s. Usage: time_limit_probe.py <floor_ckpt> [n] [max_steps]"""
import sys
sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np
from stable_baselines3 import PPO
import floor_env
from env import FALL_HEIGHT

ckpt = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 750
STAND = "/home/ubuntu/workspace/standai/checkpoints/autosave/ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip"
floor_env.SR_MAX_STEPS = max_steps
model = PPO.load(ckpt, device="cpu")
stand = PPO.load(STAND, device="cpu")
env = floor_env.FloorEnv(standready=True, crouch_frac=0.0)
first, held, ends = [], [], []
for seed in range(n):
    obs, _ = env.reset(seed=seed)
    t_reach = None
    for t in range(max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        if info.get("floor_success"):
            t_reach = t + 1
            fell = False
            for _ in range(150):
                a, _ = stand.predict(obs, deterministic=True)
                obs, *_ = env.step(a)
                if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT:
                    fell = True
                    break
            held.append(not fell)
            break
        if term or trunc:
            break
    first.append(t_reach)
    ends.append(env.step_count)
print("longest episode (steps):", max(ends))
for lim in (300, 450, 600, 750):
    if lim > max_steps:
        continue
    r = [i for i, f in enumerate(first) if f is not None and f <= lim]
    h = sum(1 for i, f in enumerate(first) if f is not None and f <= lim and held[sum(1 for g in first[:i + 1] if g is not None) - 1])
    print(f"limit {lim} steps ({lim * 0.02:.0f}s): reached {len(r)}/{n}, held by stand {h}/{n}")
print("reach step quantiles:", np.percentile([f for f in first if f], [25, 50, 75, 90]).round().tolist() if any(first) else "none")
