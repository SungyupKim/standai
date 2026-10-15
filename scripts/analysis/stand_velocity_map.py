"""How much motion can the stand policy absorb? Upright calm starts (knee >= -10 deg, torso >= 1.1 m: the 92%-hold
region) with an extra torso vertical velocity and pitch rate added; success = the 3 s gate (torso >= FALL_HEIGHT).
This is the 'stand-ready' definition for a get-up policy that hands over to the stand policy.
Usage: python scripts/analysis/stand_velocity_map.py [ckpt.zip] [n_base=30]"""
import sys; sys.path.insert(0, 'scripts')
import glob, numpy as np, mujoco
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT

ck = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.zip') else glob.glob("checkpoints/autosave/ppo_stand_v37_ladder_*_96000000_steps.zip")[0]
n_base = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[-1].isdigit() else 30
m = PPO.load(ck, device="cpu"); env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0)
ry = env.model.joint("a_root_ry").dofadr[0]
base = []
for seed in range(600):
    env.reset(seed=seed); d = env.data
    k = np.degrees(min(d.qpos[env._knee_r_qpos], d.qpos[env._knee_l_qpos])); z = d.xpos[env.a_torso_id][2]
    if k >= -10 and z >= 1.1: base.append((d.qpos.copy(), d.qvel.copy()))
    if len(base) >= n_base: break
VZ = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]; PR = [-200, -100, 0, 100, 200]
print(f"ckpt {ck.split('_')[-2]}, {len(base)} upright base starts per cell; rows = extra torso vz (m/s), cols = extra pitch rate (deg/s)")
print(f"{'vz \\ pr':>9} " + " ".join(f"{p:>6}" for p in PR))
for vz in VZ:
    row = []
    for pr in PR:
        ok = 0
        for q, v in base:
            env.reset(seed=0); env.data.qpos[:] = q; vv = v.copy(); vv[env.a_z_dof] += vz; vv[ry] += np.radians(pr); env.data.qvel[:] = vv
            mujoco.mj_forward(env.model, env.data); env._prev_com_x = None; obs = env._obs(); fell = False
            for _ in range(150):
                a, _ = m.predict(obs, deterministic=True); obs, _, te, tr, _ = env.step(a)
                if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT: fell = True; break
            ok += (not fell)
        row.append(ok / len(base))
    print(f"{vz:>9.1f} " + " ".join(f"{x:>6.0%}" for x in row), flush=True)
