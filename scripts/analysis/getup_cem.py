"""Get-up feasibility, v4. Objective changes vs v3 (which stalled at sitting height, plausibly
because cos(pitch) was multiplied in at EVERY step and so penalised the forward trunk lean that
sit-to-stand needs):
  - NO mid-trajectory pitch credit. Uprightness only required at the END (success + end term).
  - Height only counts while a foot is on the ground (kills airborne / head-supported exploits).
  - Peak foot-supported head height is rewarded regardless of pitch (lean is allowed en route).
  - Weight-shift term: while low and foot-supported, reward COM being near the feet.
  - Penalties: head touching ground, inverted torso."""
import sys; sys.path.insert(0, 'scripts')
import numpy as np, mujoco, time
from multiprocessing import Pool
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT, FRAME_SKIP

T, K, ND = 250, 20, 12
POP, ELITE, ITERS = 128, 16, 100
PITCH_OK = np.radians(40.0)
HEAD_OK = 0.9
FOOT = ('foot_l', 'foot_r', 'toe_l', 'toe_r')


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def apply_gear(env, scale=None):
    """Multiply the gear of 'a_' limb actuators (never the root ones) on top of the env's body spec.
    GEAR_SCALE=<x> picks the multiplier; GEAR_JOINTS=hip,knee restricts it to actuators whose name
    contains one of those substrings (default: all limbs). Use STANDAI_BODY_GEAR=1 for an x1 base."""
    import os
    scale = float(os.environ.get('GEAR_SCALE', '1')) if scale is None else scale
    only = [t for t in os.environ.get('GEAR_JOINTS', '').split(',') if t]
    if scale == 1.0:
        return
    m = env.model
    for i in range(m.nu):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ''
        if n.startswith('a_') and 'root' not in n and (not only or any(t in n for t in only)):
            m.actuator_gear[i, 0] *= scale
            # LocomotionEnv.reset() re-applies gear from _gear_base (gear curriculum), so scale that too
            if hasattr(env, '_gear_base'):
                env._gear_base[i] *= scale


def make_env(seed):
    env = LocomotionEnv(stand_only=True); env.set_calm_frac(0.0)
    apply_gear(env)
    env.reset(seed=seed)
    start = mujoco.MjData(env.model); mujoco.mj_copyData(start, env.model, env.data)
    return env, start


def expand(knots):
    xs = np.linspace(0, T - 1, K); t = np.arange(T)
    return np.stack([np.interp(t, xs, knots[:, j]) for j in range(ND)], axis=1)


def contacts(env):
    m, d = env.model, env.data
    names = getattr(env, '_geom_body_names', None)
    if names is None:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or '' for g in range(m.ngeom)]
        env._geom_body_names = names
    foot = head = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        n1, n2 = names[c.geom1], names[c.geom2]
        for n, o in ((n1, n2), (n2, n1)):
            if n.startswith('a_') and not (o.startswith('a_') or o.startswith('b_')):
                if n[2:] in FOOT: foot = 1.0
                if n[2:] == 'head': head = 1.0
    return foot, head


def snap(env):
    d = env.data
    feet_x = np.mean([d.site_xpos[s][0] for s in env.a_feet])
    com_x = np.sum(env._a_masses * d.xipos[env._a_bodies, 0]) / env._a_masses.sum()
    foot, head = contacts(env)
    return (d.xpos[env.a_torso_id][2], d.qpos[env.a_ry_qpos], d.xpos[env.a_head_id][2],
            foot, head, com_x - feet_x)


def rollout(env, start, acts):
    mujoco.mj_copyData(env.data, env.model, start)
    d = env.data; m = env.model
    tr = np.empty((6, T))
    for t in range(T):
        a = np.clip(acts[t], -1, 1)
        for _ in range(FRAME_SKIP):
            d.ctrl[env.a_act] = a
            d.ctrl[env.a_act[env.a_arm_act_idx]] *= env._power_scale(env.a_arm_qpos, env.a_arm_range, C['ARM'])
            d.ctrl[env.a_act[env.a_leg_act_idx]] *= env._power_scale(env.a_leg_qpos, env.a_leg_range, C['LEG'])
            d.ctrl[env.a_root_act] = 0.0
            d.ctrl[env.a_balance_act] = 0.0
            mujoco.mj_step(m, d)
        tr[:, t] = snap(env)
    return tr


def upright_stand(tr):
    z, ry, hz, foot = tr[0], tr[1], tr[2], tr[3]
    return (z >= FALL_HEIGHT) & (np.abs(wrap(ry)) < PITCH_OK) & (hz >= HEAD_OK) & (foot > 0)


def score(tr):
    z, ry, hz, foot, head, dx = tr
    phi = np.minimum(hz, 1.3) / 1.3
    supported = phi * foot                                   # height credited only with a foot down
    peak = supported.max()
    end_up = np.clip(np.cos(ry[-25:]), 0, 1) * supported[-25:]
    low = (foot > 0) & (z < 0.55)
    shift = np.where(low, np.clip(1.0 - np.abs(dx) / 0.5, 0, 1), 0.0).mean()
    stood = upright_stand(tr)[-50:].mean()
    pen = 0.5 * head.mean() + 0.5 * (np.abs(wrap(ry)) > np.radians(120)).mean()
    return peak + end_up.mean() + 0.3 * shift + 3.0 * stood - pen, stood


def solve(args):
    seed, = args
    env, start = make_env(seed)
    rng = np.random.default_rng(seed)
    mu = np.zeros((K, ND)); sd = np.full((K, ND), 0.7)
    best = (-1e9, None, None)
    for it in range(ITERS):
        cand = np.clip(mu + sd * rng.standard_normal((POP, K, ND)), -1, 1)
        res = [score(rollout(env, start, expand(c))) for c in cand]
        sc = np.array([r[0] for r in res]); idx = np.argsort(-sc)[:ELITE]
        if sc[idx[0]] > best[0]:
            best = (sc[idx[0]], cand[idx[0]].copy(), res[idx[0]])
        el = cand[idx]; mu = 0.3 * mu + 0.7 * el.mean(0); sd = np.maximum(0.25 * sd + 0.75 * el.std(0), 0.08)
    _, knots, (s, stood) = best
    env2 = LocomotionEnv(stand_only=True); env2.set_calm_frac(0.0); env2.reset(seed=seed)
    acts = expand(knots); rows = []
    for t in range(T):
        _, _, term, trunc, _ = env2.step(np.concatenate([[0.0], acts[t]]).astype(np.float32))
        rows.append(snap(env2))
        if term or trunc: break
    tr = np.array(rows).T
    full = tr.shape[1] == T
    ok = bool(full and upright_stand(tr)[-50:].all())
    sup = tr[3] > 0
    return dict(seed=seed, len=tr.shape[1], peak_sup_z=float(tr[0][sup].max()) if sup.any() else float('nan'),
                zend=float(tr[0][-20:].mean()), pend=float(np.degrees(wrap(tr[1][-1]))),
                hend=float(tr[2][-20:].mean()), foot_end=float(tr[3][-20:].mean()),
                max_lean=float(np.degrees(np.abs(wrap(tr[1])).max())),
                stood_frac=float(upright_stand(tr)[-50:].mean()), real_ok=ok, knots=knots)


C = None
def init_worker():
    global C
    import env as env_module
    C = {'ARM': env_module.ARM_MIN_POWER, 'LEG': env_module.LEG_MIN_POWER}


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [8, 16, 22, 3, 7, 11]
    print("starts:", seeds, flush=True)
    t0 = time.time()
    with Pool(6, initializer=init_worker) as p:
        out = p.map(solve, [(s,) for s in seeds])
    print(f"done in {time.time()-t0:.0f}s")
    print(f"{'seed':>4} | {'len':>3} {'peak z(foot down)':>17} {'z_end':>6} {'pitch_end':>9} {'head_end':>8} {'foot_end':>8} {'stood%(last50)':>14} | UPRIGHT-STOOD(real)")
    for r in out:
        print(f"{r['seed']:>4} | {r['len']:>3} {r['peak_sup_z']:>17.2f} {r['zend']:>6.2f} {r['pend']:>8.0f}d {r['hend']:>8.2f} {r['foot_end']:>8.2f} {r['stood_frac']*100:>13.0f}% | {r['real_ok']}")
    np.savez('checkpoints/analysis/getup_sols4.npz',
             seeds=[r['seed'] for r in out], knots=np.stack([r['knots'] for r in out]),
             real_ok=[r['real_ok'] for r in out])
