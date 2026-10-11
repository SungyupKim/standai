"""Usage: python scripts/analysis/random_explore.py [ckpt.zip]
Policy-FREE exploration on floor starts (harsh seeds, torso z0 < 0.45 m): how high does the head get if the
12 limb actions are pure random motion with temporal structure? Compares with the policy's own deterministic
behaviour (when a checkpoint is given) -- noise added around a policy mean stays anchored to what the policy
does (leg lifting), so this asks whether unanchored, sustained random motion finds anything higher."""
import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/analysis')
import numpy as np
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
import getup_cem as g
env = LocomotionEnv(stand_only=True); env.set_calm_frac(0.0)
starts = [s for s in range(300) if (env.reset(seed=s) is not None) and env.data.xpos[env.a_torso_id][2] < 0.45]
model = None
if len(sys.argv) > 1:
    from stable_baselines3 import PPO; model = PPO.load(sys.argv[1], device='cpu')

def episode(seed, act_fn):
    obs, _ = env.reset(seed=seed); heads, rows = [], []
    for t in range(250):
        obs, r, term, trunc, info = env.step(act_fn(obs, t)); s = g.snap(env); rows.append(s); heads.append(s[2])
        if term or trunc: break
    tr = np.array(rows).T
    up = bool(tr[0][-20:].mean() >= FALL_HEIGHT and abs(g.wrap(tr[1][-20:])).mean() < np.radians(40) and tr[2][-20:].mean() >= 0.9)
    return max(heads), up

def make_ar(std, rho, seed):
    rng = np.random.default_rng(seed); state = {'n': rng.standard_normal(12)}
    def f(obs, t):
        state['n'] = rho * state['n'] + np.sqrt(1 - rho ** 2) * rng.standard_normal(12)
        return np.concatenate([[0.0], np.clip(std * state['n'], -1, 1)]).astype(np.float32)
    return f

def make_hold(k, seed):                      # random +-1 target per joint, held k steps, then re-drawn
    rng = np.random.default_rng(seed); state = {'a': rng.choice([-1.0, 1.0], 12)}
    def f(obs, t):
        if t % k == 0: state['a'] = rng.choice([-1.0, 1.0], 12)
        return np.concatenate([[0.0], state['a']]).astype(np.float32)
    return f

print(f"{len(starts)} floor starts (torso z0 < 0.45 m)")
print(f"{'action source':<44}{'head>0.5m':>10}{'head>0.9m':>10}{'upright':>9}{'median max head':>17}")
def row(label, make):
    h5 = h9 = up = 0; mh = []
    for i, s in enumerate(starts):
        m, u = episode(s, make(i)); h5 += m > 0.5; h9 += m > 0.9; up += u; mh.append(m)
    n = len(starts); print(f"{label:<44}{100*h5/n:>9.0f}%{100*h9/n:>9.0f}%{up:>6d}/{n:<3}{np.median(mh):>14.2f}m")
if model is not None:
    row("policy, deterministic", lambda i: (lambda obs, t: model.predict(obs, deterministic=True)[0]))
for std, rho in ((0.5, 0.9), (1.0, 0.9), (1.0, 0.97), (1.0, 0.99), (2.0, 0.95)):
    row(f"random AR(1) std={std} rho={rho}", lambda i, std=std, rho=rho: make_ar(std, rho, i))
for k in (10, 25, 50):
    row(f"random bang-bang, held {k} steps", lambda i, k=k: make_hold(k, i))
