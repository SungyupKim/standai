"""Usage: python scripts/analysis/chain_probe.py [ckpt.zip]
Can the rise be CONTINUED from a mid-height pose? States with the head at 0.45-0.9 m are harvested from random
bang-bang episodes that started on the floor (what the training archive holds). From each, 250 steps (the env's 5 s
window) are run under: the policy alone, the policy with correlated noise, and policy-free random bang-bang held
10 steps. Reported by head-height bin: how often the head passes 0.9 m, and how often the episode ENDS genuinely upright."""
import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/analysis')
import numpy as np
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
import getup_cem as g
env = LocomotionEnv(stand_only=True); env.set_calm_frac(0.0); env.set_archive_collect(True)
rng = np.random.default_rng(1)
floor = [s for s in range(300) if (env.reset(seed=s) is not None) and env.data.xpos[env.a_torso_id][2] < 0.45]
states = []
for s in floor:
    env.reset(seed=s); a = np.zeros(13, np.float32)
    for t in range(250):
        if t % 10 == 0: a = np.concatenate([[0.0], rng.choice([-1.0, 1.0], 12)]).astype(np.float32)
        _, _, term, trunc, info = env.step(a)
        if "archive_state" in info: states.append(info["archive_state"])
        if term or trunc: break
print(f"harvested {len(states)} mid-height states from {len(floor)} random-burst floor episodes")
model = None
if len(sys.argv) > 1:
    from stable_baselines3 import PPO; model = PPO.load(sys.argv[1], device='cpu')
env.set_archive_collect(False)

def run(entry, seed, kind):
    env.set_replay_pool([(entry[0], entry[1], False)], 1.0); obs, _ = env.reset(seed=seed)
    st = {'a': None}; n = np.zeros(12); heads = []; rows = []
    for t in range(250):
        if kind == 'random':
            if t % 10 == 0: st['a'] = rng.choice([-1.0, 1.0], 12)
            a = np.concatenate([[0.0], st['a']]).astype(np.float32)
        else:
            a, _ = model.predict(obs, deterministic=True)
            if kind == 'policy+noise':
                n = 0.9 * n + np.sqrt(1 - 0.81) * rng.standard_normal(12)
                a = np.clip(a + np.concatenate([[0.0], 0.5 * n]), -1, 1).astype(np.float32)
        obs, _, term, trunc, _ = env.step(a); s = g.snap(env); rows.append(s); heads.append(s[2])
        if term or trunc: break
    tr = np.array(rows).T
    up = bool(tr[0][-20:].mean() >= FALL_HEIGHT and abs(g.wrap(tr[1][-20:])).mean() < np.radians(40) and tr[2][-20:].mean() >= 0.9)
    return max(heads) > 0.9, up

kinds = ['random'] + (['policy', 'policy+noise'] if model is not None else [])
bins = [(0.45, 0.60), (0.60, 0.75), (0.75, 0.90)]
print(f"{'start head height':<20}{'n':>4}   " + "   ".join(f"{k:<22}" for k in kinds))
for lo, hi in bins:
    sel = [(i, s) for i, s in enumerate(states) if lo <= s[2] < hi]
    cells = []
    for k in kinds:
        h9 = up = 0
        for i, s in sel:
            a, b = run(s, i, k); h9 += a; up += b
        cells.append(f"head>0.9m {100*h9/max(len(sel),1):>3.0f}%  upright {up:>2}/{len(sel):<3}")
    print(f"[{lo:.2f},{hi:.2f}) m{'':<8}{len(sel):>4}   " + "   ".join(f"{c:<22}" for c in cells))
