"""Does a policy lean on the parked opponent body (b_*) to get up? Same fixed-seed 20 s evaluation as getup_strict_eval.py, with the opponent teleported far away (--dist m along x, zero velocity) right after every reset
(dist 0 = normal). Prints standing in all of the last 25 steps / last 5 s per start side.
Usage: .venv/bin/python -I scripts/analysis/opponent_removed_eval.py --arms teacher <zip> --dist 0 40 --seed0 98000 --n 100 [--workers 4]"""
import argparse, os, sys
from multiprocessing import get_context
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
import numpy as np
from distill_controller import make_ctl, clean_env


def run(job):
    path, side, seed0, n, dist = job
    clean_env()
    import torch
    torch.set_num_threads(1)
    import mujoco
    from stable_baselines3 import PPO
    from floor_env import FloorEnv
    from controller import GetUpController
    env = FloorEnv(standready=True, level=0.7, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False, prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    m = env.model
    model = PPO.load(path, device="cpu") if path else None
    ctl = make_ctl(("prone",)) if model is None else None
    bx = m.joint("b_root_x").qposadr[0]; bxd = m.joint("b_root_x").dofadr[0]
    out = []
    for seed in range(seed0, seed0 + n):
        obs, _ = env.reset(seed=seed)
        if ctl: ctl.reset(env)
        if dist:
            env.data.qpos[bx] += dist; env.data.qvel[bxd] = 0.0; mujoco.mj_forward(env.model, env.data)
        up = []
        for _ in range(1000):
            a = ctl.act(obs, env) if model is None else model.predict(obs, deterministic=True)[0]
            obs, _, te, tr, info = env.step(a)
            if ctl: ctl.observe(env, info)
            up.append(float(env.data.xpos[env.a_torso_id][2]) >= 0.95 and abs(np.degrees(GetUpController.pitch(env))) < 30)
        up = np.array(up); out.append((up[975:].all(), up[750:].all()))
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["teacher"]); ap.add_argument("--dist", type=float, nargs="+", default=[0.0, 40.0])
    ap.add_argument("--seed0", type=int, default=98000); ap.add_argument("--n", type=int, default=100); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    k = a.n // a.workers
    with get_context("fork").Pool(a.workers) as pool:
        for arm in a.arms:
            for side in ("supine", "prone"):
                for d in a.dist:
                    r = np.concatenate(pool.map(run, [(None if arm == "teacher" else arm, side, a.seed0 + i * k, k, d) for i in range(a.workers)]))
                    print(f"{os.path.basename(arm):26s} {side:6s} opponent moved {d:4.0f} m: last 0.5 s {r[:, 0].mean():.0%} | last 5 s {r[:, 1].mean():.0%} (n={len(r)})", flush=True)


if __name__ == "__main__":
    main()
