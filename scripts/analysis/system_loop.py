"""End-to-end 'does the body end up standing?' for three controller structures, natural start distribution (supine ~80% / prone ~20%).
  A loop      : floor policy (supine v12 / prone8 by pitch sign) -> stand v37 once in the stand-ready region -> back to the floor policy when torso < 0.7 m
  B floor     : floor policies only (re-picked by pitch sign whenever the torso is below 0.7 m)
  C noloop    : floor policy -> stand v37, never back
Standing at T = last 25 steps all with torso >= 0.95 m and |pitch| < 30 deg. Also reports standing at 10 s.
Usage: system_loop.py <A|B|C> <n_seeds> [total_steps=1000]"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np
from stable_baselines3 import PPO
from floor_env import FloorEnv
from env import FALL_HEIGHT
mode, n = sys.argv[1], int(sys.argv[2]); T = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
A = "/home/ubuntu/workspace/standai/checkpoints/autosave/"
supine = PPO.load(A + "ppo_floor_v12_20261013_204028_101475000_steps.zip", device="cpu")
prone = PPO.load("/home/ubuntu/workspace/standai/checkpoints/best_prone8.zip", device="cpu")
stand = PPO.load(A + "ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip", device="cpu")
env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=10**6, prone_frac=0.0, down_timeout=False)
pitch_of = lambda d: float(((d.qpos[env.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi))
res = dict(stand_end=0, stand_10s=0, ever_in_region=0, handoffs=0, falls=0)
for seed in range(n):
    obs, _ = env.reset(seed=seed)
    cur = prone if pitch_of(env.data) <= 0 else supine
    state = "floor"; over = False; hist = []; entered = False
    for t in range(T):
        pol = stand if state == "stand" else cur
        a, _ = pol.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        z = float(env.data.xpos[env.a_torso_id][2]); p = pitch_of(env.data)
        hist.append((z, p))
        if state == "floor":
            if info.get("floor_success"):
                if mode in ("A", "C"):
                    state = "stand"; res["handoffs"] += 1
                entered = True
            elif mode in ("A", "B") and z < FALL_HEIGHT:
                cur = prone if p <= 0 else supine                 # re-pick the floor policy from the CURRENT pitch sign
        elif state == "stand" and z < FALL_HEIGHT:
            res["falls"] += 1
            if mode == "A":
                state = "floor"; cur = prone if p <= 0 else supine
            else:
                over = True                                        # C: no way back, keep stepping the stand policy
    res["ever_in_region"] += entered
    ok = lambda lo, hi: all(z >= 0.95 and abs(np.degrees(p)) < 30 for z, p in hist[lo:hi])
    res["stand_end"] += ok(T - 25, T); res["stand_10s"] += ok(500 - 25, 500)
print(f"mode {mode} ({ {'A':'loop floor<->stand','B':'floor only','C':'floor->stand, no loop'}[mode] }): {n} starts, T={T} steps ({T*0.02:.0f} s) | "
      f"standing at 10 s: {res['stand_10s']} ({res['stand_10s']/n:.0%}) | standing at the end: {res['stand_end']} ({res['stand_end']/n:.0%}) | "
      f"reached the region at least once: {res['ever_in_region']} ({res['ever_in_region']/n:.0%}) | stand-policy falls: {res['falls']}", flush=True)
