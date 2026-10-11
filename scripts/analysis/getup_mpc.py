"""Get-up feasibility via receding-horizon predictive sampling (closed-loop, replans every step).
Physics is the env's own (rollout fidelity vs env.step was verified to 0.0000)."""
import sys; sys.path.insert(0, 'scripts')
sys.path.insert(0, 'scripts/analysis')
import numpy as np, mujoco, time
from multiprocessing import Pool
import getup_cem as g   # reuse make_env, snap, upright_stand, wrap, FOOT etc.
from env import FALL_HEIGHT, FRAME_SKIP

T = 250
H = 25            # horizon (control steps) = 0.5 s
N = 128           # samples per replan
KN = 6            # noise knots over the horizon
SIGMA = 0.45
ND = g.ND


def apply_ctrl(env, a):
    d = env.data
    for _ in range(FRAME_SKIP):
        d.ctrl[env.a_act] = a
        d.ctrl[env.a_act[env.a_arm_act_idx]] *= env._power_scale(env.a_arm_qpos, env.a_arm_range, g.C['ARM'])
        d.ctrl[env.a_act[env.a_leg_act_idx]] *= env._power_scale(env.a_leg_qpos, env.a_leg_range, g.C['LEG'])
        d.ctrl[env.a_root_act] = 0.0
        d.ctrl[env.a_balance_act] = 0.0
        mujoco.mj_step(env.model, d)


def horizon_cost(tr):
    z, ry, hz, foot, head, dx = tr
    phi = np.minimum(hz, 1.3) / 1.3
    sup = phi * foot
    up = np.clip(np.cos(ry), 0, 1)
    bonus_up = (z > 0.5) * up * sup
    low = (foot > 0) & (z < 0.55)
    shift = np.where(low, np.clip(1.0 - np.abs(dx) / 0.5, 0, 1), 0.0)
    pen = 0.5 * head + 0.5 * (np.abs(g.wrap(ry)) > np.radians(120))
    r = sup + 0.5 * bonus_up + 0.3 * shift - pen
    stood = g.upright_stand(tr).mean()
    return r.mean() + 2.0 * sup[-1] + 1.0 * bonus_up[-1] + 3.0 * stood


def run_seed(args):
    seed, = args
    env, cur = g.make_env(seed)
    m = env.model
    rng = np.random.default_rng(seed)
    plan = np.zeros((H, ND))
    xs_k = np.linspace(0, H - 1, KN); tt = np.arange(H)
    log = []; acts_log = []
    for step in range(T):
        # candidates: shifted previous plan + smooth noise (candidate 0 = plan itself)
        noise = rng.standard_normal((N, KN, ND)) * SIGMA
        noise[0] = 0.0
        cands = np.empty((N, H, ND))
        for j in range(ND):
            for i in range(N):
                cands[i, :, j] = plan[:, j] + np.interp(tt, xs_k, noise[i, :, j])
        cands = np.clip(cands, -1, 1)
        best, best_c = None, -1e9
        for i in range(N):
            mujoco.mj_copyData(env.data, m, cur)
            tr = np.empty((6, H))
            for t in range(H):
                apply_ctrl(env, cands[i, t]); tr[:, t] = g.snap(env)
            c = horizon_cost(tr)
            if c > best_c: best_c, best = c, i
        plan = cands[best]
        a0 = plan[0].copy()
        mujoco.mj_copyData(env.data, m, cur)
        apply_ctrl(env, a0)
        mujoco.mj_copyData(cur, m, env.data)
        log.append(g.snap(env)); acts_log.append(a0)
        plan = np.vstack([plan[1:], plan[-1:]])      # shift
    tr = np.array(log).T; acts = np.array(acts_log)
    # verify by open-loop replay of the logged actions through the real env.step
    env2 = g.LocomotionEnv(stand_only=True); env2.set_calm_frac(0.0); g.apply_gear(env2); env2.reset(seed=seed)
    rows = []
    for t in range(T):
        _, _, term, trunc, _ = env2.step(np.concatenate([[0.0], acts[t]]).astype(np.float32))
        rows.append(g.snap(env2))
        if term or trunc: break
    tr2 = np.array(rows).T
    ok = bool(tr2.shape[1] == T and g.upright_stand(tr2)[-50:].all())
    sup = tr2[3] > 0
    first = int(np.argmax(g.upright_stand(tr2))) if g.upright_stand(tr2).any() else -1
    return dict(seed=seed, len=tr2.shape[1], replay_match=float(np.abs(tr2[0] - tr[0][:tr2.shape[1]]).max()),
                peak_sup_z=float(tr2[0][sup].max()) if sup.any() else float('nan'),
                zend=float(tr2[0][-20:].mean()), pend=float(np.degrees(g.wrap(tr2[1][-1]))),
                hend=float(tr2[2][-20:].mean()), stood_frac=float(g.upright_stand(tr2)[-50:].mean()),
                first_upright=first, real_ok=ok, acts=acts)


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [8, 16, 22, 3, 7, 11]
    print("starts:", seeds, flush=True)
    t0 = time.time()
    with Pool(6, initializer=g.init_worker) as p:
        out = p.map(run_seed, [(s,) for s in seeds])
    print(f"done in {time.time()-t0:.0f}s")
    print(f"{'seed':>4} | {'len':>3} {'peak z(foot)':>12} {'z_end':>6} {'pitch_end':>9} {'head_end':>8} {'stood%':>6} {'1st upright t':>13} {'replay err':>10} | UPRIGHT-STOOD(real)")
    for r in out:
        print(f"{r['seed']:>4} | {r['len']:>3} {r['peak_sup_z']:>12.2f} {r['zend']:>6.2f} {r['pend']:>8.0f}d {r['hend']:>8.2f} {r['stood_frac']*100:>5.0f}% {r['first_upright']:>13} {r['replay_match']:>10.4f} | {r['real_ok']}")
    np.savez('checkpoints/analysis/getup_mpc_sols_gear.npz',
             seeds=[r['seed'] for r in out], acts=np.stack([r['acts'] for r in out]), real_ok=[r['real_ok'] for r in out])
