"""Floor policy ALONE after reaching the stand-ready region: three criteria on the same reached states.
 gate3s : torso never below FALL_HEIGHT (0.7 m) during the first 150 steps (the criterion used for every 'held' number)
 end3s  : torso >= 0.7 m at step 150 (what the viewer printed after 3 s)
 end4s  : torso >= 0.7 m at step 200 (what the viewer printed after 4 s)
Usage: floor_only_hold.py <floor_ckpt> <n> <prone_frac>"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np
from stable_baselines3 import PPO
from floor_env import FloorEnv
from env import FALL_HEIGHT
ck, n, pf = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
floor = PPO.load(ck, device="cpu")
env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=450, prone_frac=pf, down_timeout=False)
reached = gate = end3 = end4 = 0
for seed in range(n):
    obs, _ = env.reset(seed=seed); got = False
    for t in range(450):
        a, _ = floor.predict(obs, deterministic=True); obs, _, term, trunc, info = env.step(a)
        if info.get("floor_success"): got = True; break
        if term or trunc: break
    if not got: continue
    reached += 1; ok_gate = True; z150 = z200 = 0.0
    for k in range(200):
        a, _ = floor.predict(obs, deterministic=True); obs, *_ = env.step(a)
        z = float(env.data.xpos[env.a_torso_id][2])
        if k < 150 and z < FALL_HEIGHT: ok_gate = False
        if k == 149: z150 = z
        if k == 199: z200 = z
    gate += ok_gate; end3 += z150 >= FALL_HEIGHT; end4 += z200 >= FALL_HEIGHT
print(f"{ck.split('/')[-1]} (prone_frac {pf}): starts {n}, reached {reached} | floor policy alone: gate3s {gate} ({gate/max(reached,1):.0%}) | end3s {end3} ({end3/max(reached,1):.0%}) | end4s {end4} ({end4/max(reached,1):.0%})", flush=True)
