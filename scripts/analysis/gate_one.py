"""Usage: python scripts/analysis/gate_one.py <ckpt.zip>   (run from repo root)
Deterministic 3s survival gate on seeds 0..99: calm / full-dist / harsh-only, nominal body."""
import sys; sys.path.insert(0, 'scripts')
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
ckpt = sys.argv[1]
model = PPO.load(ckpt, device='cpu')
def gate(calm_frac, n=100):
    env = LocomotionEnv(stand_only=True)
    if calm_frac is not None: env.set_calm_frac(calm_frac)
    ok = 0
    for seed in range(n):
        obs, _ = env.reset(seed=seed); fell = False
        for t in range(150):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT: fell = True; break
            if term or trunc: break
        if not fell: ok += 1
    return ok
steps = ckpt.split('_')[-2]
print(f"{int(steps):>10} calm={gate(1.0):>3} full={gate(None):>3} harsh={gate(0.0):>3}", flush=True)
