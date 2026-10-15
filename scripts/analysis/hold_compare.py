"""After the floor policy reaches the stand-ready region: does the STAND policy (v37) or the FLOOR policy itself keep the torso above
FALL_HEIGHT for 3 s? Same reached states for both. Usage: hold_compare.py <floor_ckpt> <n> <prone_frac>"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np
from stable_baselines3 import PPO
from floor_env import FloorEnv
from env import FALL_HEIGHT
ck, n, pf = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
STAND = "/home/ubuntu/workspace/standai/checkpoints/autosave/ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip"
floor, stand = PPO.load(ck, device="cpu"), PPO.load(STAND, device="cpu")
def hold(env, pol, obs):
    for _ in range(150):
        a, _ = pol.predict(obs, deterministic=True); obs, *_ = env.step(a)
        if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT: return False
    return True
reached = by_stand = by_floor = either = 0
env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=450, prone_frac=pf, down_timeout=False)
for seed in range(n):
    # run the floor policy to the first success, snapshot the physics state, then replay the continuation twice
    obs, _ = env.reset(seed=seed); got = False
    for t in range(450):
        a, _ = floor.predict(obs, deterministic=True); obs, _, term, trunc, info = env.step(a)
        if info.get("floor_success"): got = True; break
        if term or trunc: break
    if not got: continue
    reached += 1
    import mujoco
    snap = mujoco.MjData(env.model); mujoco.mj_copyData(snap, env.model, env.data); obs0 = obs.copy(); sc = env.step_count
    hs = hold(env, stand, obs0)
    mujoco.mj_copyData(env.data, env.model, snap); env.step_count = sc
    hf = hold(env, floor, obs0)
    by_stand += hs; by_floor += hf; either += (hs or hf)
print(f"{ck.split('/')[-1]} (prone_frac {pf}): starts {n}, reached {reached} | held 3 s by stand v37: {by_stand} ({by_stand/max(reached,1):.0%} of reached) | "
      f"by the floor policy itself: {by_floor} ({by_floor/max(reached,1):.0%}) | either: {either} ({either/max(reached,1):.0%})", flush=True)
