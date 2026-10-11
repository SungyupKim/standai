"""Usage: python scripts/analysis/explore_probe.py <ckpt.zip>
Does MORE or TEMPORALLY-CORRELATED action noise help a policy find its way up from the floor?
Floor starts (torso z0 < 0.45 m, harsh seeds 0..299). Per noise setting: how often the head ever
exceeds 0.5 m / 0.9 m and how often the episode ends genuinely upright. mult scales the policy's own
learned std; rho is the one-step autocorrelation of the noise (0 = white = what PPO samples)."""
import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/analysis')
import numpy as np
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
import getup_cem as g
model = PPO.load(sys.argv[1], device='cpu')
std = model.policy.log_std.exp().detach().numpy()
env = LocomotionEnv(stand_only=True); env.set_calm_frac(0.0)
starts = []
for seed in range(300):
    env.reset(seed=seed)
    if env.data.xpos[env.a_torso_id][2] < 0.45: starts.append(seed)
print(f"{len(starts)} floor starts; policy std mean {std.mean():.2f}")
print(f"{'noise':<26}{'head>0.5m':>10}{'head>0.9m':>10}{'upright end':>13}{'max head (median)':>19}")
for mult, rho, label in ((0, 0, 'deterministic'), (1, 0, 'x1 white (as trained)'), (2, 0, 'x2 white'), (3, 0, 'x3 white'),
                          (1, 0.9, 'x1 correlated rho=.9'), (2, 0.9, 'x2 correlated rho=.9'), (3, 0.95, 'x3 correlated rho=.95')):
    rng = np.random.default_rng(0); h5 = h9 = up = 0; mh = []
    for seed in starts:
        obs, _ = env.reset(seed=seed); n = np.zeros_like(std); heads = []; rows = []
        for t in range(250):
            a, _ = model.predict(obs, deterministic=True)
            n = rho * n + np.sqrt(1 - rho ** 2) * rng.standard_normal(std.shape)
            a = np.clip(a + mult * std * n, -1, 1).astype(np.float32)
            obs, r, term, trunc, info = env.step(a); s = g.snap(env); rows.append(s); heads.append(s[2])
            if term or trunc: break
        tr = np.array(rows).T
        h5 += max(heads) > 0.5; h9 += max(heads) > 0.9; mh.append(max(heads))
        up += bool(tr[0][-20:].mean() >= FALL_HEIGHT and abs(g.wrap(tr[1][-20:])).mean() < np.radians(40) and tr[2][-20:].mean() >= 0.9)
    n_ = len(starts); print(f"{label:<26}{100*h5/n_:>9.0f}%{100*h9/n_:>9.0f}%{up:>9d}/{n_:<3}{np.median(mh):>18.2f}m")
