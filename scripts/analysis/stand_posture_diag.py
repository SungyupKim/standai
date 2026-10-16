"""What does a stand-policy checkpoint do on calm starts and on strict hand-off states? Torso height / pitch / head after 3 s (the v39 collapse: flat on the floor, 0/300). Usage: stand_posture_diag.py <ckpt>"""
import sys, glob, os; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np, mujoco
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
ck = sys.argv[1]
m = PPO.load(ck, device="cpu"); print(os.path.basename(ck))
env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0)
zs, mins, heads, pitch = [], [], [], []
for seed in range(60):
    obs, _ = env.reset(seed=seed); z0 = float(env.data.xpos[env.a_torso_id][2]); lo = z0
    for t in range(150):
        a, _ = m.predict(obs, deterministic=True); obs, *_ = env.step(a)
        z = float(env.data.xpos[env.a_torso_id][2]); lo = min(lo, z)
    zs.append(z); mins.append(lo); heads.append(float(env.data.xpos[env.a_head_id][2]))
    pitch.append(float(np.degrees((env.data.qpos[env.a_ry_qpos] + np.pi) % (2*np.pi) - np.pi)))
zs, mins = np.array(zs), np.array(mins)
print(f"calm starts (60): torso at 3 s median {np.median(zs):.2f} m (p10 {np.percentile(zs,10):.2f}, p90 {np.percentile(zs,90):.2f}); min torso median {np.median(mins):.2f}; "
      f"never below 0.7: {(mins >= FALL_HEIGHT).sum()}; |pitch| at 3 s median {np.median(np.abs(pitch)):.0f} deg; head z median {np.median(heads):.2f}")
d = np.load("/home/ubuntu/workspace/standai/checkpoints/eval_strict_all.npz", allow_pickle=True)
env2 = LocomotionEnv(stand_only=True); held = 0; endz = []
for i in range(len(d["qpos"])):
    env2.reset(seed=i); env2.data.qpos[:], env2.data.qvel[:] = d["qpos"][i], d["qvel"][i]; mujoco.mj_forward(env2.model, env2.data)
    env2._prev_com_x = None; obs = env2._obs(); ok = True
    for t in range(150):
        a, _ = m.predict(obs, deterministic=True); obs, *_ = env2.step(a)
        z = float(env2.data.xpos[env2.a_torso_id][2])
        if z < FALL_HEIGHT: ok = False
    endz.append(z); held += ok
print(f"hand-off eval set (300): never below 0.7 m for 3 s: {held}; torso at 3 s median {np.median(endz):.2f} m")
