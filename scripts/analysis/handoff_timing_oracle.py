"""Upper bound for choosing WHEN to hand over: after the floor policy first reaches the stand-ready region (step t0) keep running it, and every 5 steps
in [t0-20, t0+100] (torso >= 0.8 m) test 'would the stand policy hold 3 s if it took over here?' (state snapshot, v37, deterministic).
Reports per start: held at t0 exactly (what the system does now), held at ANY candidate step (oracle), and the offsets of the first catchable moment.
Usage: handoff_timing_oracle.py <floor_ckpt> <n_starts> <prone_frac>"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np, mujoco
from stable_baselines3 import PPO
from floor_env import FloorEnv
from env import FALL_HEIGHT
ck, n, pf = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
STAND = "/home/ubuntu/workspace/standai/checkpoints/autosave/ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip"
floor, stand = PPO.load(ck, device="cpu"), PPO.load(STAND, device="cpu")
env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=10**6, prone_frac=pf, down_timeout=False)

def holds(snap, obs):
    mujoco.mj_copyData(env.data, env.model, snap)
    o = obs
    for _ in range(150):
        a, _ = stand.predict(o, deterministic=True); o, *_ = env.step(a)
        if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT:
            return False
    return True

reached = at_t0 = oracle = 0; offsets = []; n_cand = []
from collections import defaultdict
per_off = defaultdict(lambda: [0, 0])      # offset -> [tested, held]
for seed in range(n):
    obs, _ = env.reset(seed=seed)
    hist = []            # (snapshot, obs, torso z) for every step so far (the 20 steps before entry are needed)
    t0 = None
    for t in range(450 + 120):
        a, _ = floor.predict(obs, deterministic=True); obs, _, term, trunc, info = env.step(a)
        z = float(env.data.xpos[env.a_torso_id][2])
        snap = mujoco.MjData(env.model); mujoco.mj_copyData(snap, env.model, env.data)
        hist.append((snap, obs.copy(), z))
        if t0 is None and info.get("floor_success"):
            t0 = t
        if t0 is not None and t >= t0 + 100:
            break
        if t0 is None and t >= 450:
            break
    if t0 is None:
        continue
    reached += 1
    cands = [k for k in range(max(0, t0 - 20), min(len(hist), t0 + 101)) if (k - t0) % 5 == 0 and hist[k][2] >= 0.8]
    n_cand.append(len(cands))
    ok_t0 = holds(*hist[t0][:2]); at_t0 += ok_t0
    first = None
    for k in cands:
        ok = ok_t0 if k == t0 else holds(*hist[k][:2])
        per_off[k - t0][0] += 1; per_off[k - t0][1] += ok
        if ok and first is None:
            first = k - t0
    if first is not None:
        oracle += 1; offsets.append(first)
print(f"{ck.split('/')[-1]} (prone_frac {pf}): starts {n}, reached region {reached} | held when handed over at entry: {at_t0} ({at_t0/max(reached,1):.0%}) | "
      f"catchable at SOME candidate moment: {oracle} ({oracle/max(reached,1):.0%}) | median candidates/start {np.median(n_cand):.0f} | "
      f"first catchable offset (steps from entry) median {np.median(offsets) if offsets else float('nan'):.0f}, share handed over later than entry "
      f"{(sum(o != 0 for o in offsets) / max(len(offsets), 1)):.0%}", flush=True)
print("hold rate by hand-over offset (steps from the first entry into the region; tested / held):")
print("  " + " | ".join(f"{o:+d}: {h}/{t} ({h / t:.0%})" for o, (t, h) in sorted(per_off.items()) if t >= 8), flush=True)
