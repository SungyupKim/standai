"""Can ANY controller keep the body up for 3 s from a given state? (diagnostic, not a teacher)

Receding-horizon predictive sampling with the env's own physics (same machinery as getup_mpc.py), but started from
arbitrary saved states and judged by the 3 s gate: torso never below FALL_HEIGHT for 150 control steps.
Groups: A = floor-policy hand-off states, B = crouched calm starts (knee < -40 deg), C = upright calm starts (control).
Usage: python scripts/analysis/mpc_hold.py [n_per_group=24] [workers=12]
"""
import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/analysis')
import glob, os, re, time
import numpy as np, mujoco
from multiprocessing import Pool
import getup_cem as g
import getup_mpc as gm
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT

T, H, N, KN, SIGMA = 150, 25, 128, 6, 0.45
ND = g.ND


def cost(tr):
    z, ry, hz, foot, head, dx = tr
    up = np.clip(np.cos(ry), 0, 1)
    return (np.minimum(z, 1.15).mean() + 1.0 * up.mean() + 1.0 * min(z[-1], 1.15) + 1.0 * up[-1]
            - 6.0 * (z < FALL_HEIGHT).mean() - 0.5 * head.mean())


def run(job):
    label, idx, qpos, qvel = job
    env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0); env.reset(seed=0)
    env.data.qpos[:], env.data.qvel[:] = qpos, qvel
    mujoco.mj_forward(env.model, env.data)
    cur = mujoco.MjData(env.model); mujoco.mj_copyData(cur, env.model, env.data)
    rng = np.random.default_rng(idx); plan = np.zeros((H, ND)); xs = np.linspace(0, H - 1, KN); tt = np.arange(H)
    zmin, held = 9.0, True
    for step in range(T):
        noise = rng.standard_normal((N, KN, ND)) * SIGMA; noise[0] = 0.0
        cands = np.empty((N, H, ND))
        for j in range(ND):
            for i in range(N):
                cands[i, :, j] = plan[:, j] + np.interp(tt, xs, noise[i, :, j])
        cands = np.clip(cands, -1, 1)
        best, bc = 0, -1e9
        for i in range(N):
            mujoco.mj_copyData(env.data, env.model, cur); tr = np.empty((6, H))
            for t in range(H):
                gm.apply_ctrl(env, cands[i, t]); tr[:, t] = g.snap(env)
            c = cost(tr)
            if c > bc: bc, best = c, i
        plan = cands[best]
        mujoco.mj_copyData(env.data, env.model, cur); gm.apply_ctrl(env, plan[0]); mujoco.mj_copyData(cur, env.model, env.data)
        z = float(env.data.xpos[env.a_torso_id][2]); zmin = min(zmin, z)
        if z < FALL_HEIGHT: held = False; break
        plan = np.vstack([plan[1:], plan[-1:]])
    return dict(label=label, idx=idx, held=held, zmin=zmin, steps=step + 1)


def build_jobs(n):
    eval_set = np.load("checkpoints/handoff_eval_v4.npz", allow_pickle=True)
    jobs = [("A hand-off", i, eval_set["qpos"][i], eval_set["qvel"][i]) for i in range(n)]
    env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0); crouch, upright = [], []
    for seed in range(400):
        env.reset(seed=seed); d = env.data
        k = np.degrees(min(d.qpos[env._knee_r_qpos], d.qpos[env._knee_l_qpos]))
        rec = (d.qpos.copy(), d.qvel.copy(), seed)
        if k < -40 and len(crouch) < n: crouch.append(rec)
        if k >= -10 and len(upright) < n: upright.append(rec)
    jobs += [("B crouched calm", s, q, v) for q, v, s in crouch] + [("C upright calm", s, q, v) for q, v, s in upright]
    return jobs


def policy_hold(jobs):
    from stable_baselines3 import PPO
    m = PPO.load(glob.glob("checkpoints/autosave/ppo_stand_v37_ladder_*_96000000_steps.zip")[0], device="cpu")
    env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0); out = {}
    for label, idx, qpos, qvel in jobs:
        env.reset(seed=0); env.data.qpos[:], env.data.qvel[:] = qpos, qvel; mujoco.mj_forward(env.model, env.data)
        env._prev_com_x = None; obs = env._obs(); ok = True
        for _ in range(T):
            a, _ = m.predict(obs, deterministic=True); obs, _, _, _, _ = env.step(a)
            if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT: ok = False; break
        out.setdefault(label, []).append(ok)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    jobs = build_jobs(n); t0 = time.time()
    pol = policy_hold(jobs)
    res = []
    with Pool(workers, initializer=g.init_worker) as p:
        for r in p.imap_unordered(run, jobs, chunksize=1):      # print as they finish so a killed run still leaves data
            res.append(r); print(f"  [{len(res)}/{len(jobs)}] {r['label']} #{r['idx']}: {'HELD' if r['held'] else 'fell'} (min torso {r['zmin']:.2f})", flush=True)
    print(f"done in {time.time()-t0:.0f}s   (hold = torso stays >= {FALL_HEIGHT} m for {T} control steps = 3 s)")
    for label in dict.fromkeys(j[0] for j in jobs):
        r = [x for x in res if x["label"] == label]; held = sum(x["held"] for x in r)
        pp = pol[label]
        print(f"{label:>16}: MPC oracle holds {held}/{len(r)} ({held/len(r):.0%}) | stand policy v37@96M holds {sum(pp)}/{len(pp)} ({sum(pp)/len(pp):.0%})")
