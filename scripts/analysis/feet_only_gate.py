"""Alternative definition of 'standing': BOTH feet touch the floor and NO other body part does.
Compares it with the current gate (torso never below FALL_HEIGHT 0.7 m for 3 s). Usage: python scripts/analysis/feet_only_gate.py"""
import sys; sys.path.insert(0, 'scripts')
import glob, os, re, numpy as np
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv, _FOOT_BODIES
from floor_env import FloorEnv, SR_MAX_STEPS
from env import FALL_HEIGHT
import mujoco

def floor_contact_bodies(env):
    m, d = env.model, env.data
    names = getattr(env, '_geom_body_names', None)
    if names is None:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or '' for g in range(m.ngeom)]; env._geom_body_names = names
    s = set()
    for i in range(d.ncon):
        c = d.contact[i]; n1, n2 = names[c.geom1], names[c.geom2]
        for n, o in ((n1, n2), (n2, n1)):
            if n.startswith('a_') and not (o.startswith('a_') or o.startswith('b_')): s.add(n[2:])
    return s

def feet_only(env):
    s = floor_contact_bodies(env)
    left = any(b in s for b in ('foot_l', 'toe_l')); right = any(b in s for b in ('foot_r', 'toe_r'))
    return left and right and s <= set(_FOOT_BODIES), s

step = lambda p: int(re.search(r"_(\d+)_steps", p).group(1))
stand = PPO.load(glob.glob("checkpoints/autosave/ppo_stand_v37_ladder_*_96000000_steps.zip")[0], device="cpu")
# A) stand policy on calm starts, 300 seeds: old gate vs feet-only gate
env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0); old = new = both = 0; leaks = {}
for seed in range(300):
    obs,_ = env.reset(seed=seed); ok_old = True; ok_new = True; frac = 0
    for t in range(150):
        a,_ = stand.predict(obs, deterministic=True); obs,_,te,tr,_ = env.step(a)
        if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT: ok_old = False
        fo, s = feet_only(env)
        if not fo:
            ok_new = False
            for b in (s - set(_FOOT_BODIES)): leaks[b] = leaks.get(b, 0) + 1
        if te or tr: break
    old += ok_old; new += ok_new; both += (ok_old and ok_new)
print(f"A) stand policy v37@96M, 300 calm starts, 3 s: old gate (torso>=0.7 m) {old}/300 ({old/300:.0%}) | feet-only gate (both feet, nothing else on the floor, EVERY step) {new}/300 ({new/300:.0%}) | both {both}")
print("   non-foot parts that touched the floor in feet-only failures (episode-steps):", dict(sorted(leaks.items(), key=lambda kv: -kv[1])[:6]))
# B) floor v8 + stand hand-over on lying starts: how many end standing under each definition
fl = PPO.load("checkpoints/autosave/ppo_floor_v8_20261013_064037_94275000_steps.zip", device="cpu")
fenv = FloorEnv(standready=True, crouch_frac=0.0); N = 200; reach = hold_old = hold_new = feet_first = 0
for seed in range(N):
    obs,_ = fenv.reset(seed=seed); got = False
    for t in range(SR_MAX_STEPS):
        a,_ = fl.predict(obs, deterministic=True); obs,_,te,tr,info = fenv.step(a)
        if info.get("floor_success"): got = True; break
        if te or tr: break
    if not got: continue
    reach += 1; ok_old = True; ok_new = True
    for t in range(150):
        a,_ = stand.predict(obs, deterministic=True); obs,*_ = fenv.step(a)
        if fenv.data.xpos[fenv.a_torso_id][2] < FALL_HEIGHT: ok_old = False
        if not feet_only(fenv)[0]: ok_new = False
    hold_old += ok_old; hold_new += ok_new
print(f"B) lying starts ({N}): floor policy reaches stand-ready {reach}; stand policy then holds 3 s: old gate {hold_old} | feet-only gate {hold_new}")
