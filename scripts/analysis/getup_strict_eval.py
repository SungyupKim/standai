"""Is a high end-to-end score real? Re-measures a student zip (or the rule-based teacher) on fresh seeds with STRICTER and richer criteria than the benchmark's 'last 25 steps upright':
standing in ALL of the last 100 / 250 steps (2 s / 5 s), time-averaged standing fraction over the last 10 s, torso-height std over the last 5 s (hopping / bouncing shows up here), mean joint speed in the last 5 s.
Usage: .venv/bin/python -I scripts/analysis/getup_strict_eval.py --arms teacher <zip> --seed0 98000 --n 200 [--workers 8]"""
import argparse, os, sys
from multiprocessing import get_context
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
import numpy as np
from distill_controller import make_ctl, clean_env, CKPTS


def run(job):
    path, side, seed0, n = job
    clean_env()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from floor_env import FloorEnv
    from controller import GetUpController
    env = FloorEnv(standready=True, level=0.7, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False, prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    model = PPO.load(path, device="cpu") if path else None
    ctl = make_ctl(("prone",)) if model is None else None
    rows = []
    for seed in range(seed0, seed0 + n):
        obs, _ = env.reset(seed=seed)
        if ctl: ctl.reset(env)
        z, up, jv = [], [], []
        for _ in range(1000):
            a = ctl.act(obs, env) if model is None else model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(a)
            if ctl: ctl.observe(env, info)
            zz = float(env.data.xpos[env.a_torso_id][2]); z.append(zz)
            up.append(zz >= 0.95 and abs(np.degrees(GetUpController.pitch(env))) < 30)
            jv.append(float(np.abs(env.data.qvel[3:17]).mean()))
        up = np.array(up); z = np.array(z)
        rows.append((up[975:].all(), up[900:].all(), up[750:].all(), up[500:].mean(), z[750:].std(), float(np.mean(jv[750:])), float(z[750:].mean()), float(env.data.xpos[env.a_torso_id][2]) >= 0.0))
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["teacher"]); ap.add_argument("--seed0", type=int, default=98000)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    k = a.n // a.workers
    with get_context("fork").Pool(a.workers) as pool:
        for arm in a.arms:
            for side in ("supine", "prone"):
                r = np.concatenate(pool.map(run, [(None if arm == "teacher" else arm, side, a.seed0 + i * k, k) for i in range(a.workers)]))
                print(f"{os.path.basename(arm):34s} {side:6s} n={len(r)}: last 0.5 s {r[:,0].mean():.0%} | last 2 s {r[:,1].mean():.0%} | last 5 s {r[:,2].mean():.0%} | standing fraction of the last 10 s {r[:,3].mean():.0%} | "
                      f"torso z std (last 5 s) {r[:,4].mean():.3f} m | mean joint speed {r[:,5].mean():.2f} | mean torso z {r[:,6].mean():.2f} m", flush=True)


if __name__ == "__main__":
    main()
