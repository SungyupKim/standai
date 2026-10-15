"""Does the floor policy slow down as it gets high? Speed profile by torso height on lying starts.
For every step of every episode, bin |pitch rate| (deg/s) and |vz| (m/s) by torso height; also the 'braking' summary: the
share of steps above 1.0 m whose pitch rate is inside the stand-ready tolerance (< 100 deg/s).
Usage: python scripts/analysis/speed_profile.py [ckpt.zip] [n=100]"""
import sys; sys.path.insert(0, 'scripts')
import glob, re, numpy as np
from stable_baselines3 import PPO
from floor_env import FloorEnv, SR_MAX_STEPS
step = lambda p: int(re.search(r"_(\d+)_steps", p).group(1))
ck = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.zip') else max(glob.glob("checkpoints/autosave/ppo_floor_v*_steps.zip"), key=lambda p: (p.split('_')[2], step(p)))
n = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 100
m = PPO.load(ck, device="cpu"); env = FloorEnv(standready=True, crouch_frac=0.0)
BINS = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.08), (1.08, 9)]
pr = {b: [] for b in BINS}; vz = {b: [] for b in BINS}; ok = 0
for seed in range(n):
    obs, _ = env.reset(seed=seed)
    for t in range(SR_MAX_STEPS):
        a, _ = m.predict(obs, deterministic=True); obs, _, te, tr, info = env.step(a)
        z = float(env.data.xpos[env.a_torso_id][2])
        for b in BINS:
            if b[0] <= z < b[1]:
                pr[b].append(abs(np.degrees(float(env.data.qvel[env._ry_dof])))); vz[b].append(abs(float(env.data.qvel[env.a_z_dof])))
        if te or tr: break
    ok += bool(info.get("floor_success"))
print(f"{ck.split('/')[-1]}  ({n} lying seeds, reached stand-ready {ok})")
print(f"{'torso height':>13} | steps | |pitch rate| deg/s: median / p90 | share < 100 | |vz| m/s: median / p90")
for b in BINS:
    p, v = np.array(pr[b]), np.array(vz[b])
    if len(p) == 0: continue
    print(f"{b[0]:>5.2f}-{(b[1] if b[1] < 9 else float('inf')):<6.2f} | {len(p):>5} | {np.median(p):>9.0f} / {np.percentile(p, 90):>5.0f}        | {np.mean(p < 100):>9.0%}   | {np.median(v):.2f} / {np.percentile(v, 90):.2f}")
hi = np.array(pr[(1.0, 1.08)] + pr[(1.08, 9)])
print(f"braking summary: above 1.0 m, {np.mean(hi < 100):.0%} of steps have |pitch rate| < 100 deg/s")
