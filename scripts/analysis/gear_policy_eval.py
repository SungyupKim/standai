"""Usage: python scripts/analysis/gear_policy_eval.py <gear_scale> [ckpt.zip]
Calm gate (seeds 0..299) of a policy on a body whose limb gears are multiplied by gear_scale.
Note: the policy was trained at x1, so >x1.5 mixes torque effects with policy mismatch."""
import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'scripts/analysis')
import glob, os, numpy as np, mujoco
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
import getup_cem as g
scale = float(sys.argv[1])
ckpt = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith('.zip') else max(glob.glob('checkpoints/autosave/ppo_stand_v*_*_steps.zip'), key=os.path.getmtime)
model = PPO.load(ckpt, device='cpu')
env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0); g.apply_gear(env, scale)
kn, ok = [], []
for seed in range(300):
    obs,_ = env.reset(seed=seed); d = env.data
    kn.append(np.degrees(min(d.qpos[env._knee_r_qpos], d.qpos[env._knee_l_qpos]))); fell=False
    for t in range(150):
        a,_ = model.predict(obs, deterministic=True)
        obs,r,term,trunc,info = env.step(a)
        if d.xpos[env.a_torso_id][2] < FALL_HEIGHT: fell=True; break
        if term or trunc: break
    ok.append(not fell)
kn, ok = np.array(kn), np.array(ok)
edges = [-180,-60,-40,-25,-10,5]
row = " ".join(f"{100*ok[(kn>=lo)&(kn<hi)].mean():>4.0f}%" for lo,hi in zip(edges[:-1],edges[1:]))
print(f"gear x{scale:<4} overall {ok.sum():>3}/300 | by knee [<-60, -60..-40, -40..-25, -25..-10, >-10]: {row}  ({os.path.basename(ckpt).split('_')[-2]})", flush=True)
