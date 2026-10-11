"""Usage: python scripts/analysis/harsh_diag.py [ckpt.zip]
Harsh starts (seeds 0..299): who ends standing, height gain, split by start height."""
import sys; sys.path.insert(0, 'scripts')
import glob, os, numpy as np
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
ckpt = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.zip') else max(glob.glob('checkpoints/autosave/ppo_stand_v*_*_steps.zip'), key=os.path.getmtime)
print('ckpt:', ckpt)
model = PPO.load(ckpt, device='cpu')
env = LocomotionEnv(stand_only=True); env.set_calm_frac(0.0)
T = 450; R = []
for seed in range(300):
    obs, _ = env.reset(seed=seed); d = env.data
    zs = [d.xpos[env.a_torso_id][2]]; acts = []; ended = None
    for t in range(T):
        a, _ = model.predict(obs, deterministic=True); acts.append(a)
        obs, r, term, trunc, info = env.step(a)
        zs.append(d.xpos[env.a_torso_id][2])
        if term or trunc: ended = t; break
    zs = np.array(zs); acts = np.array(acts)
    first = int(np.argmax(zs >= FALL_HEIGHT)) if (zs >= FALL_HEIGHT).any() else None
    R.append(dict(z0=zs[0], zmax=zs.max(), zend=zs[-20:].mean(), n=len(zs), first=first,
                  act_sd=float(acts.std(axis=0).mean()), z150=zs[min(150, len(zs)-1)], ended=ended))
z0 = np.array([r['z0'] for r in R]); low = z0 < FALL_HEIGHT
zmax = np.array([r['zmax'] for r in R]); zend = np.array([r['zend'] for r in R])
z150 = np.array([r['z150'] for r in R]); sd = np.array([r['act_sd'] for r in R])
n = np.array([r['n'] for r in R])
print(f"early-terminated episodes: {(n < T+1).sum()}/300")
print(f"ended standing @450 (last-20 mean>=thr): all {(zend>=FALL_HEIGHT).sum()}/300; below-start {(zend[low]>=FALL_HEIGHT).sum()}/{low.sum()}; above-start {(zend[~low]>=FALL_HEIGHT).sum()}/{(~low).sum()}")
print(f"ended standing @150 (z at step150>=thr), below-start: {(z150[low]>=FALL_HEIGHT).sum()}/{low.sum()}")
gain = (zmax - z0)[low]
print("below-start: height gain (zmax-z0) pctiles 10/25/50/75/90 (m):", np.percentile(gain,[10,25,50,75,90]).round(2))
print("below-start: final height pctiles 10/50/90 (m):", np.percentile(zend[low],[10,50,90]).round(2))
print(f"below-start: gain<0.05m (basically didn't rise): {(gain<0.05).sum()}/{low.sum()}")
print("action temporal sd (movement proxy): below-start mean %.3f  above-start mean %.3f" % (sd[low].mean(), sd[~low].mean()))
bins=[0,0.15,0.3,0.45,0.6,0.7]
for lo,hi in zip(bins[:-1],bins[1:]):
    m=(z0>=lo)&(z0<hi)
    if m.sum(): print(f"  start z [{lo:.2f},{hi:.2f}) n={m.sum():>3}  ended standing={(zend[m]>=FALL_HEIGHT).sum():>3}  mean final z={zend[m].mean():.2f}")
