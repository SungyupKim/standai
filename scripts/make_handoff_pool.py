"""Collects hand-off states from the floor get-up policy (floor_env.FloorEnv) for the stand policy to start from.

Each saved state is (qpos, qvel) at the instant the floor policy first reaches the hand-off pose. Seeds are
disjoint from the evaluation seeds (0..199 are the evaluation hand-offs), and half of the episodes sample the
floor policy stochastically so the pool is not one state per start.

Usage: make_handoff_pool.py [--ckpt FLOOR_CKPT] [--n 1500] [--out ../checkpoints/handoff_pool_v4.npz]
"""
import argparse
import glob
import os

import numpy as np
from stable_baselines3 import PPO

from floor_env import FloorEnv

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=None)
p.add_argument("--n", type=int, default=1500)
p.add_argument("--out", default="../checkpoints/handoff_pool_v4.npz")
p.add_argument("--first-seed", type=int, default=1000)
p.add_argument("--deterministic", action="store_true", help="never sample actions (evaluation sets)")
args = p.parse_args()

ckpt = args.ckpt or max(glob.glob("../checkpoints/autosave/ppo_floor_v4_*_steps.zip"), key=os.path.getmtime)
model = PPO.load(ckpt, device="cpu")
env = FloorEnv()
qpos, qvel, heads, tries = [], [], [], 0
seed = args.first_seed
while len(qpos) < args.n and tries < args.n * 6:
    obs, _ = env.reset(seed=seed)
    stochastic = (seed % 2 == 1) and not args.deterministic   # odd seeds: sample actions, even seeds: deterministic
    tries += 1; seed += 1
    for _ in range(250):
        action, _ = model.predict(obs, deterministic=not stochastic)
        obs, _, term, trunc, info = env.step(action)
        if info.get("floor_success"):
            q, v, h = info["handoff_state"]
            qpos.append(q); qvel.append(v); heads.append(h)
            break
        if trunc:
            break
np.savez(args.out, qpos=np.array(qpos), qvel=np.array(qvel), head=np.array(heads), floor_ckpt=os.path.basename(ckpt))
print(f"{len(qpos)} hand-off states from {tries} starts ({len(qpos) / tries:.0%}) -> {args.out} (floor ckpt {os.path.basename(ckpt)})")
