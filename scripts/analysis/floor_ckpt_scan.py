"""Scans many checkpoints of a floor-policy run on the same fixed seeds (hand-off region reached within --max-steps, deterministic) to pick the best one with a sample large
enough to see differences (the policy's outcome is chaotic, so +-5 points at 100 seeds hides real differences).
Usage: STANDAI_NO_RECOVERY_GATE=1 .venv/bin/python -I scripts/analysis/floor_ckpt_scan.py --glob 'checkpoints/autosave/ppo_floor_v12_*_steps.zip' [--stride 4] [--n 200] [--seed0 25000] [--side supine] [--level 0.7]"""
import argparse, glob, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
import numpy as np


def reach(args):
    ck, side, n, seed0, level, max_steps = args
    from stable_baselines3 import PPO
    from floor_env import FloorEnv
    m = PPO.load(ck, device="cpu")
    env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=max_steps, down_timeout=False, level=level, speed_pen=0.08, speed_vz_w=0.25,
                   prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    ok = []
    for s in range(seed0, seed0 + n):
        obs, _ = env.reset(seed=s); r = False
        for t in range(max_steps):
            a, _ = m.predict(obs, deterministic=True); obs, _, te, tr, info = env.step(a)
            if info.get("floor_success"): r = True; break
        ok.append(r)
    return ck, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed0", type=int, default=25000)
    ap.add_argument("--side", default="supine"); ap.add_argument("--level", type=float, default=0.7)
    ap.add_argument("--max-steps", type=int, default=450); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=ROOT + "checkpoints/analysis/floor_ckpt_scan.json")
    a = ap.parse_args()
    key = lambda f: int(re.search(r"_(\d+)_steps", f).group(1))
    files = sorted(glob.glob(a.glob), key=key)[::a.stride]
    print(f"{len(files)} checkpoints x {a.n} seeds ({a.seed0}..), {a.side}, level {a.level}", flush=True)
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for ck, ok in ex.map(reach, [(f, a.side, a.n, a.seed0, a.level, a.max_steps) for f in files]):
            res[ck] = ok; print(f"  {key(ck) / 1e6:8.3f}M  reached {np.mean(ok):.1%}", flush=True)
    json.dump({k: v for k, v in res.items()}, open(a.out, "w"))
    rows = sorted(res.items(), key=lambda kv: -np.mean(kv[1]))
    print("top 5:", [(f"{key(k) / 1e6:.3f}M", f"{np.mean(v):.1%}") for k, v in rows[:5]])
    m = np.array([np.mean(v) for v in res.values()]); print(f"mean {m.mean():.1%}, std over checkpoints {m.std():.1%}, binomial std at n={a.n} would be {np.sqrt(m.mean() * (1 - m.mean()) / a.n):.1%}")
