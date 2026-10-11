"""Usage: python scripts/analysis/calm_knee_table.py [ckpt.zip]   (default: newest autosave)
Calm-start gate on seeds 0..299, success rate binned by initial knee bend and torso height."""
import sys; sys.path.insert(0, 'scripts')
import glob, os, numpy as np
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
ckpt = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.zip') else max(glob.glob('checkpoints/autosave/ppo_stand_v*_*_steps.zip'), key=os.path.getmtime)
print('ckpt:', ckpt)
model = PPO.load(ckpt, device='cpu')
env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0)
km0s, z0s, ok = [], [], []
for seed in range(300):
    obs, _ = env.reset(seed=seed); d = env.data
    z0s.append(d.xpos[env.a_torso_id][2])
    km0s.append(np.degrees(min(d.qpos[env._knee_r_qpos], d.qpos[env._knee_l_qpos])))
    fell = False
    for t in range(150):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        if d.xpos[env.a_torso_id][2] < FALL_HEIGHT: fell = True; break
        if term or trunc: break
    ok.append(not fell)
km0s, z0s, ok = np.array(km0s), np.array(z0s), np.array(ok)
print(f"overall ok {ok.sum()}/300")
def table(name, x, edges):
    print(f"\n{name}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum(): print(f"  [{lo:>7.2f},{hi:>7.2f}) n={m.sum():>3}  ok={ok[m].sum():>3} ({100*ok[m].mean():.0f}%)")
table("initial min-knee qpos (deg; 0=straight, more negative=more bent)", km0s, [-180,-90,-60,-40,-25,-10,5])
table("initial torso z (m)", z0s, [0,0.9,1.0,1.08,1.12,1.16,2])
