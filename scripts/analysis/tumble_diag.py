"""Usage: python scripts/analysis/tumble_diag.py <ckpt.zip>   (STANDAI_BODY_GEAR=1 for pre-v28 checkpoints)
Harsh starts (seeds 0..299), deterministic policy, 250 steps. Measures how much the torso ROTATES
(total |d pitch| / 2pi = full turns) and whether the episode ENDS genuinely upright (torso z >= 0.7,
|pitch| < 40deg, head z >= 0.9) rather than merely torso-z >= FALL_HEIGHT, which tumbling can satisfy."""
import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/analysis')
import numpy as np
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
import getup_cem as g

model = PPO.load(sys.argv[1], device='cpu')
env = LocomotionEnv(stand_only=True); env.set_calm_frac(0.0)
turns, inv, z_ok, up_ok, below, below_up, below_zok = [], [], [], [], [], [], []
for seed in range(300):
    obs, _ = env.reset(seed=seed)
    rows = [g.snap(env)]
    for t in range(250):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        rows.append(g.snap(env))
        if term or trunc: break
    tr = np.array(rows).T                       # z, pitch, head z, foot, head contact, dx
    z, ry, hz = tr[0], tr[1], tr[2]
    turns.append(np.abs(np.diff(ry)).sum() / (2 * np.pi))
    inv.append((np.abs(g.wrap(ry)) > np.radians(120)).mean())
    zok = z[-20:].mean() >= FALL_HEIGHT
    upok = bool(zok and np.abs(g.wrap(ry[-20:])).mean() < np.radians(40) and hz[-20:].mean() >= 0.9)
    z_ok.append(zok); up_ok.append(upok); below.append(z[0] < FALL_HEIGHT)
turns, inv, z_ok, up_ok, below = map(np.array, (turns, inv, z_ok, up_ok, below))
print(f"{sys.argv[1].split('/')[-1]}")
print(f"  full turns of the torso per episode: mean {turns.mean():.2f}, median {np.median(turns):.2f}, >=1 turn in {100*(turns>=1).mean():.0f}% of episodes")
print(f"  time spent inverted (|pitch|>120deg): mean {100*inv.mean():.0f}% of the episode")
print(f"  ends with torso z>=0.7 : all {z_ok.sum()}/300 | started below 0.7m: {z_ok[below].sum()}/{below.sum()}")
print(f"  ends genuinely UPRIGHT : all {up_ok.sum()}/300 | started below 0.7m: {up_ok[below].sum()}/{below.sum()}")
