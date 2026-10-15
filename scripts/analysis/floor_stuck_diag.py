"""Where does the floor policy get stuck on the v6+ stand-ready target?
For every episode (fixed seeds, deterministic) it records which of the 6 stand-ready conditions were EVER met and, at the
step of highest torso, which condition(s) failed. Usage: python scripts/analysis/floor_stuck_diag.py [ckpt] [n=200]"""
import sys; sys.path.insert(0, 'scripts')
import glob, os, re, numpy as np
from stable_baselines3 import PPO
from floor_env import (FloorEnv, SR_TORSO_Z, SR_MAX_KNEE_BEND_DEG, SR_MAX_PITCH_DEG, SR_MAX_VZ, SR_MAX_PITCH_RATE, SR_MAX_STEPS)
step = lambda p: int(re.search(r"_(\d+)_steps", p).group(1))
ck = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.zip') else max(glob.glob("checkpoints/autosave/ppo_floor_v8_*_steps.zip"), key=step)
n = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 200
m = PPO.load(ck, device="cpu"); env = FloorEnv(standready=True)
COND = ["foot", "torso>=1.08", "knee>=-20", "|pitch|<20", "|vz|<1.0", "|pr|<100"]
rows = {"lying": [], "crouch": []}
for seed in range(n):
    obs, _ = env.reset(seed=seed); kind = env._start_kind; ever = np.zeros(6, bool); best = None; zmax = -1
    for t in range(SR_MAX_STEPS):
        a, _ = m.predict(obs, deterministic=True); obs, _, te, tr, info = env.step(a)
        foot, _ = env._getup_contacts(); z = float(env.data.xpos[env.a_torso_id][2]); knee = env._knee_min_deg()
        pitch = abs(np.degrees((env.data.qpos[env.a_ry_qpos] + np.pi) % (2*np.pi) - np.pi)); vz = abs(float(env.data.qvel[env.a_z_dof])); pr = abs(np.degrees(float(env.data.qvel[env._ry_dof])))
        c = np.array([foot > 0, z >= SR_TORSO_Z, knee >= -SR_MAX_KNEE_BEND_DEG, pitch < SR_MAX_PITCH_DEG, vz < SR_MAX_VZ, pr < SR_MAX_PITCH_RATE]); ever |= c
        if z > zmax: zmax, best = z, (c.copy(), knee, pitch, vz, pr)
        if te or tr: break
    rows[kind].append(dict(ok=bool(info.get("floor_success")), ever=ever, zmax=zmax, best=best))
print(f"ckpt v8 @{step(ck)/1e6:.2f}M, {n} fixed seeds, deterministic")
for kind, R in rows.items():
    ok = [r for r in R if r["ok"]]; bad = [r for r in R if not r["ok"]]
    print(f"\n{kind}: {len(R)} starts, reached stand-ready {len(ok)}")
    print("  condition EVER met (all episodes): " + ", ".join(f"{c} {np.mean([r['ever'][i] for r in R]):.0%}" for i, c in enumerate(COND)))
    if bad:
        print(f"  failures ({len(bad)}): peak torso median {np.median([r['zmax'] for r in bad]):.2f} m (need {SR_TORSO_Z}); at the peak-torso step the condition that FAILED:")
        fails = np.array([~r["best"][0] for r in bad]).mean(0)
        print("     " + ", ".join(f"{c} {fails[i]:.0%}" for i, c in enumerate(COND)))
        print(f"     median at peak torso: knee {np.median([r['best'][1] for r in bad]):.0f} deg, |pitch| {np.median([r['best'][2] for r in bad]):.0f} deg, |vz| {np.median([r['best'][3] for r in bad]):.2f} m/s, |pitch rate| {np.median([r['best'][4] for r in bad]):.0f} deg/s")
        only = [int(np.where(~r['best'][0])[0][0]) for r in bad if (~r['best'][0]).sum() == 1]
        print(f"     failures missing exactly ONE condition at peak torso: {len(only)}/{len(bad)}" + (" -> " + ", ".join(f"{COND[i]}:{only.count(i)}" for i in sorted(set(only))) if only else ""))
