"""Usage: python scripts/analysis/reward_by_height.py [ckpt.zip] [--ground-shaping | --getup | --knee-gate]
Mean per-step reward and each term, binned by torso height, on harsh starts. Exposed the
"lying flat pays more than a partial rise" valley."""
import sys; sys.path.insert(0, 'scripts')
import glob, os, numpy as np
from collections import defaultdict
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
ckpt = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.zip') else max(glob.glob('checkpoints/autosave/ppo_stand_v*_*_steps.zip'), key=os.path.getmtime)
print('ckpt:', ckpt)
model = PPO.load(ckpt, device='cpu')
env = LocomotionEnv(stand_only=True, ground_shaping='--ground-shaping' in sys.argv, getup_reward='--getup' in sys.argv, knee_gate='--knee-gate' in sys.argv); env.set_calm_frac(0.0)
KEYS = ["height","stability","stance","knee_avoid","recovery","jerk","velocity","gait","brace","com","stand_streak","knee_straight","torso_upright","ground_height","getup_shift","getup_pen"]
edges = [0,0.2,0.35,0.5,0.7,0.9,2.0]
acc = defaultdict(lambda: defaultdict(float)); cnt = defaultdict(int); tot = defaultdict(float)
for seed in range(300):
    obs, _ = env.reset(seed=seed)
    for t in range(450):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        z = env.data.xpos[env.a_torso_id][2]
        b = next(i for i in range(len(edges)-1) if edges[i] <= z < edges[i+1]) if z < edges[-1] else len(edges)-2
        cnt[b] += 1; tot[b] += r
        bd = info.get("reward_breakdown", {})
        for k in KEYS: acc[b][k] += bd.get(k, 0.0)
        if term or trunc: break
print(f"{'torso z':>12} {'steps':>7} {'TOTAL':>7} | " + " ".join(f"{k[:8]:>8}" for k in KEYS))
for b in range(len(edges)-1):
    if not cnt[b]: continue
    n = cnt[b]
    print(f"[{edges[b]:.2f},{edges[b+1]:.2f}) {n:>7} {tot[b]/n:>7.2f} | " + " ".join(f"{acc[b][k]/n:>8.2f}" for k in KEYS))
