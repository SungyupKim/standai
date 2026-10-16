"""Calm gate (3 s, deterministic, n fixed seeds) of one stand checkpoint under the CURRENT environment code. Used to check that the environment still
reproduces an old checkpoint's known number (v21 @82.425M: calm 33/100 at gear x1, 43/100 at x1.5). Usage: [STANDAI_BODY_GEAR=1] [STANDAI_GATE_KNEE_FLOOR=-78] gate_calm_n.py <ckpt> [n]"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
m = PPO.load(sys.argv[1], device="cpu"); n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0); ok = 0
import os
_kf = os.environ.get("STANDAI_GATE_KNEE_FLOOR")       # the knee-angle floor training applies to calm episodes (KneeScaffoldCallback, -50 -> -80 deg)
if _kf: env.set_min_knee_deg(float(_kf))
for seed in range(n):
    obs, _ = env.reset(seed=seed); fell = False
    for t in range(150):
        a, _ = m.predict(obs, deterministic=True); obs, r, term, trunc, info = env.step(a)
        if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT: fell = True; break
        if term or trunc: break
    ok += (not fell)
print(f"calm gate {ok}/{n}  (body gear env STANDAI_BODY_GEAR={os.environ.get('STANDAI_BODY_GEAR', 'default x1.5')}, knee floor {_kf or 'none'})")
