"""At the moment a prone floor policy reaches the stand-ready region: which features separate states the stand policy holds 3 s from
states it does not? Usage: handoff_features.py <floor_ckpt> [n]"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np
from stable_baselines3 import PPO
from floor_env import FloorEnv
from env import FALL_HEIGHT
ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
STAND = "/home/ubuntu/workspace/standai/checkpoints/autosave/ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip"
model, stand = PPO.load(ck, device="cpu"), PPO.load(STAND, device="cpu")
env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=450, prone_frac=1.0, down_timeout=False)
wj = env.model.joint("a_waist").qposadr[0]
names = ["torso z", "knee min", "pitch", "vz", "pitch rate", "waist", "head x - torso x", "head z", "reach step"]
rows, held = [], []
for seed in range(n):
    obs, _ = env.reset(seed=seed)
    for t in range(450):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        if info.get("floor_success"):
            d = env.data
            f = [float(d.xpos[env.a_torso_id][2]), env._knee_min_deg(),
                 float(np.degrees((d.qpos[env.a_ry_qpos] + np.pi) % (2*np.pi) - np.pi)), float(d.qvel[env.a_z_dof]),
                 float(np.degrees(d.qvel[env._ry_dof])), float(np.degrees(d.qpos[wj])),
                 float(d.xpos[env.a_head_id][0] - d.xpos[env.a_torso_id][0]), float(d.xpos[env.a_head_id][2]), t]
            fell, tmin = False, 9.0
            for _ in range(150):
                a, _ = stand.predict(obs, deterministic=True)
                obs, *_ = env.step(a)
                z = float(env.data.xpos[env.a_torso_id][2]); tmin = min(tmin, z)
                if z < FALL_HEIGHT: fell = True; break
            rows.append(f); held.append(not fell)
            break
        if term or trunc: break
r, h = np.array(rows), np.array(held)
print(f"{ck.split('_')[-2]} steps: reached {len(r)}/{n}, held {h.sum()} ({h.mean():.0%} of reached)")
print(f"{'feature':>18} | {'held mean (p10..p90)':>26} | {'not held mean (p10..p90)':>26}")
for i, nm in enumerate(names):
    a, b = r[h, i], r[~h, i]
    fmt = lambda x: f"{x.mean():+8.2f} ({np.percentile(x,10):+.1f}..{np.percentile(x,90):+.1f})" if len(x) else "-"
    print(f"{nm:>18} | {fmt(a):>26} | {fmt(b):>26}")
