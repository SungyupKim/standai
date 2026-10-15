"""At the highest-torso step of each prone-start episode: torso pitch (+ = head toward -x = leaning back, assuming the body faces +x),
waist angle, knee, torso/head height. Usage: peak_posture.py <ckpt> [n]"""
import sys
sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np
from stable_baselines3 import PPO
from floor_env import FloorEnv
ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
model = PPO.load(ck, device="cpu")
env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=450, prone_frac=1.0, down_timeout=False)
wj = env.model.joint("a_waist").qposadr[0]
rows = []
for seed in range(n):
    obs, _ = env.reset(seed=seed)
    best = None
    for t in range(450):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        z = float(env.data.xpos[env.a_torso_id][2])
        if best is None or z > best[0]:
            pitch = float(np.degrees((env.data.qpos[env.a_ry_qpos] + np.pi) % (2*np.pi) - np.pi))
            best = (z, pitch, float(np.degrees(env.data.qpos[wj])), env._knee_min_deg(), float(env.data.xpos[env.a_head_id][2]), bool(env._getup_contacts()[0]), t)
        if term or trunc: break
    rows.append(best)
r = np.array([b[:5] for b in rows], dtype=float)
print(f"{ck.split('/')[-1]}: {n} prone starts, state at the highest-torso step")
for lo, hi, lab in ((0.9, 9, "torso >= 0.9 m"), (1.0, 9, "torso >= 1.0 m")):
    s = r[(r[:, 0] >= lo) & (r[:, 0] < hi)]
    if len(s) == 0: continue
    print(f"{lab}: {len(s)} episodes | pitch deg median {np.median(s[:,1]):+.0f} (p10 {np.percentile(s[:,1],10):+.0f}, p90 {np.percentile(s[:,1],90):+.0f}) | "
          f"leaning back (pitch > +20): {(s[:,1] > 20).sum()}, leaning forward (< -20): {(s[:,1] < -20).sum()}, upright (|p|<=20): {(abs(s[:,1]) <= 20).sum()} | "
          f"waist deg median {np.median(s[:,2]):+.0f} (p10 {np.percentile(s[:,2],10):+.0f}, p90 {np.percentile(s[:,2],90):+.0f}) | knee median {np.median(s[:,3]):+.0f}")
print("waist range deg: -20 .. +45")
s = r[r[:, 0] >= 0.9]
if len(s):
    back = s[s[:, 1] > 20]; up = s[abs(s[:, 1]) <= 20]
    if len(back): print(f"  back-leaning group waist median {np.median(back[:,2]):+.0f}")
    if len(up): print(f"  upright group waist median {np.median(up[:,2]):+.0f}")
