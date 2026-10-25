"""Robustness: after the get-up (t = 500..800, random time) an external horizontal force of +-F newtons is applied to the torso for 5 steps (0.1 s); standing in all of the last 25 steps is counted.
Compares the rule-based controller with a single-network zip on fixed seeds. Usage: .venv/bin/python -I scripts/analysis/push_eval.py --arms teacher <zip> --force 0 150 300 --seed0 98000 --n 100"""
import argparse, os, sys
from multiprocessing import get_context
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
import numpy as np
from distill_controller import make_ctl, clean_env


def run(job):
    path, side, seed0, n, force = job
    clean_env()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from floor_env import FloorEnv
    from controller import GetUpController
    env = FloorEnv(standready=True, level=0.7, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False, prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    model = PPO.load(path, device="cpu") if path else None
    ctl = make_ctl(("prone",)) if model is None else None
    rng = np.random.default_rng(seed0)
    tid = env.a_torso_id
    out = []
    for seed in range(seed0, seed0 + n):
        obs, _ = env.reset(seed=seed)
        if ctl: ctl.reset(env)
        t_push = int(rng.integers(500, 800)); sgn = 1.0 if rng.random() < 0.5 else -1.0
        up = []
        for t in range(1000):
            env.data.xfrc_applied[tid, 0] = sgn * force if t_push <= t < t_push + 5 else 0.0
            a = ctl.act(obs, env) if model is None else model.predict(obs, deterministic=True)[0]
            obs, _, te, tr, info = env.step(a)
            if ctl: ctl.observe(env, info)
            up.append(float(env.data.xpos[tid][2]) >= 0.95 and abs(np.degrees(GetUpController.pitch(env))) < 30)
        env.data.xfrc_applied[:] = 0.0
        out.append(all(up[975:]))
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["teacher"]); ap.add_argument("--force", type=float, nargs="+", default=[0.0, 150.0, 300.0])
    ap.add_argument("--seed0", type=int, default=98000); ap.add_argument("--n", type=int, default=100); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    k = a.n // a.workers
    with get_context("fork").Pool(a.workers) as pool:
        for arm in a.arms:
            for side in ("supine", "prone"):
                row = []
                for f in a.force:
                    r = np.concatenate(pool.map(run, [(None if arm == "teacher" else arm, side, a.seed0 + i * k, k, f) for i in range(a.workers)]))
                    row.append(f"{f:.0f} N {r.mean():.0%}")
                print(f"{os.path.basename(arm):26s} {side:6s} standing at the end after a 0.1 s push: " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
