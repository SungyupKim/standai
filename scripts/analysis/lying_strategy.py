"""Usage: python scripts/analysis/lying_strategy.py <ckpt.zip>   (STANDAI_BODY_GEAR=1 for pre-v28 checkpoints)
While the torso is low (z < 0.5) on harsh starts: which body parts touch the floor, how high the knees
go, how fast the hips swing, and how much 'knee_avoid' reward is being paid -- split by whether the
episode ends genuinely upright. Tests the 'leg swing / momentum instead of arms' impression."""
import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/analysis')
import numpy as np, mujoco
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
import getup_cem as g

model = PPO.load(sys.argv[1], device='cpu')
env = LocomotionEnv(stand_only=True); env.set_calm_frac(0.0)
m, d = env.model, env.data
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM if False else mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or '' for i in range(m.ngeom)]
GROUPS = {'arms': ('upper_arm', 'forearm'), 'thighs/shins': ('thigh', 'shin'), 'feet': ('foot', 'toe'),
          'head': ('head',), 'torso/chest': ('torso', 'chest')}
hips = [int(m.joint(n).dofadr[0]) for n in ('a_hip_r', 'a_hip_l')]
rows = []
for seed in range(300):
    obs, _ = env.reset(seed=seed)
    if d.xpos[env.a_torso_id][2] >= FALL_HEIGHT: continue          # only starts that begin on the ground
    touch = {k: 0 for k in GROUPS}; low = 0; kz = []; hs = []; ka = []; zs = []; ry = [d.qpos[env.a_ry_qpos]]
    for t in range(250):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        z = d.xpos[env.a_torso_id][2]; zs.append(z); ry.append(d.qpos[env.a_ry_qpos])
        if z < 0.5:
            low += 1
            seen = set()
            for i in range(d.ncon):
                c = d.contact[i]
                for gi, gj in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    n, o = names[gi], names[gj]
                    if n.startswith('a_') and not (o.startswith('a_') or o.startswith('b_')):
                        for k, keys in GROUPS.items():
                            if any(s in n for s in keys): seen.add(k)
            for k in seen: touch[k] += 1
            kz.append(min(d.xanchor[env.a_knee_r_id][2], d.xanchor[env.a_knee_l_id][2]))
            hs.append(np.mean([abs(d.qvel[h]) for h in hips]))
            ka.append(info.get('reward_breakdown', {}).get('knee_avoid', 0.0))
        if term or trunc: break
    if low < 20: continue
    up = bool(np.mean(zs[-20:]) >= FALL_HEIGHT and abs(g.wrap(np.array(ry[-20:]))).mean() < np.radians(40))
    rows.append(dict(up=up, low=low, **{k: v / low for k, v in touch.items()}, kz=np.mean(kz), kz_max=np.max(kz),
                     hip=np.mean(hs), ka=np.mean(ka), turns=np.abs(np.diff(ry)).sum() / (2 * np.pi)))
def show(label, rs):
    if not rs: print(f"  {label}: n=0"); return
    f = lambda k: np.mean([r[k] for r in rs])
    print(f"  {label} (n={len(rs)}): floor contact while low -> " + ", ".join(f"{k} {100*f(k):.0f}%" for k in GROUPS)
          + f"\n      lowest knee height mean {f('kz'):.2f}m (peak {f('kz_max'):.2f}m) | hip swing speed {f('hip'):.1f} rad/s | knee_avoid reward {f('ka'):.2f}/step | torso turns {f('turns'):.2f}")
print(sys.argv[1].split('/')[-1], f"- harsh starts that begin on the ground, torso z<0.5 phases")
show("all", rows); show("ended upright", [r for r in rows if r['up']]); show("did not", [r for r in rows if not r['up']])
