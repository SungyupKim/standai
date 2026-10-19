"""Why do lying starts NOT reach the stand-ready region? Deterministic floor policy, fixed seeds 0..n-1, lying starts only.
Per seed: peak torso height, best number of stand-ready conditions (of 6) met at one step, which condition is missing
at the highest-torso step, and the start pitch sign (face up/down mapping unverified).
Classes of misses: A never rose (peak torso < 0.7 m), B rose partway (0.7-1.08 m), C reached 1.08 m but never all 6 conditions;
within C: near miss = 5/6 conditions at some step in the last 100 steps.
Usage: floor_miss_classify.py <floor_ckpt> [n] [max_steps] [prone_frac] [supine_frac] [level]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from collections import Counter
from stable_baselines3 import PPO
from floor_env import (FloorEnv, SR_TORSO_Z, SR_MAX_KNEE_BEND_DEG, SR_MAX_PITCH_DEG, SR_MAX_VZ, SR_MAX_PITCH_RATE)

ckpt = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 450
NAMES = ["foot", "torso>=1.08", "knee", "pitch<20", "|vz|<1", "|pr|<100"]
model = PPO.load(ckpt, device="cpu")
prone_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0     # 1.0 = prone-side starts only
supine_frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0    # 1.0 = supine-side starts only
level = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=max_steps, prone_frac=prone_frac, supine_frac=supine_frac, level=level)
rows = []
for seed in range(n):
    obs, _ = env.reset(seed=seed)
    p0 = float((env.data.qpos[env.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi)
    t0 = float(env.data.xpos[env.a_torso_id][2])
    peak, peak_conds, best, best_t, reached = 0.0, None, 0, 0, False
    for t in range(max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        torso = float(env.data.xpos[env.a_torso_id][2])
        pitch = float((env.data.qpos[env.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi)
        c = [bool(env._getup_contacts()[0]), torso >= SR_TORSO_Z, env._knee_min_deg() >= -SR_MAX_KNEE_BEND_DEG,
             abs(pitch) < np.deg2rad(SR_MAX_PITCH_DEG), abs(float(env.data.qvel[env.a_z_dof])) < SR_MAX_VZ,
             abs(float(np.degrees(env.data.qvel[env._ry_dof]))) < SR_MAX_PITCH_RATE]
        if torso > peak:
            peak, peak_conds = torso, c
        if sum(c) >= best:
            best, best_t = sum(c), t
        if info.get("floor_success"):
            reached = True
            break
        if term or trunc:
            break
    rows.append(dict(seed=seed, reached=reached, peak=peak, peak_conds=peak_conds, best=best, best_t=best_t,
                     steps=env.step_count, p0=p0, t0=t0))

miss = [r for r in rows if not r["reached"]]
print(f"{ckpt.split('/')[-1]}: reached {n - len(miss)}/{n}, missed {len(miss)}")
A = [r for r in miss if r["peak"] < 0.7]
B = [r for r in miss if 0.7 <= r["peak"] < SR_TORSO_Z]
C = [r for r in miss if r["peak"] >= SR_TORSO_Z]
near = [r for r in C if r["best"] >= 5 and r["best_t"] >= max_steps - 100]
print(f"A never rose (peak torso < 0.7 m): {len(A)}")
print(f"B rose partway (0.7-1.08 m):       {len(B)}   (median peak {np.median([r['peak'] for r in B]):.2f} m)" if B else "B: 0")
print(f"C reached 1.08 m, never all 6:     {len(C)}   of which 5/6 conditions in the last 100 steps: {len(near)}")
if C:
    cnt = Counter()
    for r in C:
        for nm, ok in zip(NAMES, r["peak_conds"]):
            cnt[nm] += (not ok)
    print("   condition failing at the highest-torso step (C):", dict(cnt))
print("peak torso quantiles over misses (10/50/90):", np.percentile([r["peak"] for r in miss], [10, 50, 90]).round(2).tolist())
for lab, sel in (("start pitch > 0", lambda r: r["p0"] > 0), ("start pitch <= 0", lambda r: r["p0"] <= 0)):
    g = [r for r in rows if sel(r)]
    if g:
        print(f"{lab}: {len(g)} starts, reached {sum(r['reached'] for r in g)} ({sum(r['reached'] for r in g) / len(g):.0%}), "
              f"misses of class A: {sum(1 for r in g if (not r['reached']) and r['peak'] < 0.7)}")
