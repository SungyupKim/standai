"""Collects STRICT stand-ready hand-off states (floor_env.FloorEnv(standready=True)) from a floor policy, for training the stand policy.

One state per start: (qpos, qvel) at the first step the floor policy reaches the stand-ready region. Seeds default to 1000+, disjoint from the
evaluation seeds 0..399. Odd seeds sample the policy's actions (noise -> a wider spread of states), even seeds are deterministic.
Usage: make_strict_pool.py --ckpt FLOOR_CKPT --n 2000 --prone-frac 1.0 --first-seed 1000 --out ../checkpoints/pool_part.npz"""
import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO

from floor_env import FloorEnv

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--n", type=int, default=2000)
p.add_argument("--prone-frac", type=float, default=0.0)
p.add_argument("--first-seed", type=int, default=1000)
p.add_argument("--out", required=True)
p.add_argument("--deterministic", action="store_true", help="never sample actions (evaluation sets)")
a = p.parse_args()

model = PPO.load(a.ckpt, device="cpu")
env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=450, prone_frac=a.prone_frac, down_timeout=False)
qpos, qvel, heads, tries, seed = [], [], [], 0, a.first_seed
t0 = time.time()
while len(qpos) < a.n and tries < a.n * 5:
    obs, _ = env.reset(seed=seed)
    stochastic = seed % 2 == 1 and not a.deterministic
    seed += 1; tries += 1
    if tries % 200 == 0:
        el = time.time() - t0
        eta = el * (a.n - len(qpos)) / max(len(qpos), 1)
        print(f"[progress] {len(qpos)}/{a.n} states from {tries} starts ({len(qpos) / tries:.0%}), {el / 60:.0f} min elapsed, ~{eta / 60:.0f} min left", flush=True)
    for _ in range(450):
        action, _ = model.predict(obs, deterministic=not stochastic)
        obs, _, term, trunc, info = env.step(action)
        if info.get("floor_success"):
            q, v, h = info["handoff_state"]
            qpos.append(q); qvel.append(v); heads.append(h)
            break
        if trunc:
            break
np.savez(a.out, qpos=np.array(qpos), qvel=np.array(qvel), head=np.array(heads), floor_ckpt=os.path.basename(a.ckpt))
print(f"{len(qpos)} states from {tries} starts ({len(qpos) / max(tries, 1):.0%}) -> {a.out} ({os.path.basename(a.ckpt)})", flush=True)
