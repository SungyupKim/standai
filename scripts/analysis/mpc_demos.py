"""MPC demonstrations (obs, action) from hand-off pool states, for imitation of the stand policy.
Runs mpc_handoff.mpc_state on a random sample of states of a pool npz (default: checkpoints/pool_big_all.npz, half supine / half prone, which are
the first / second half of that file), keeps the trajectories where MPC caught the stand (benchmark criterion), and saves them incrementally.
Usage: .venv/bin/python -I scripts/analysis/mpc_demos.py --n 1200 --workers 15 --out checkpoints/mpc_demos_a.npz"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
sys.path.insert(0, ROOT + "scripts/analysis")
os.environ["MPC_SAVE_DEMOS"] = "1"
import numpy as np
import mpc_handoff as mh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=ROOT + "checkpoints/pool_big_all.npz")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--samples", type=int, default=96)
    a = ap.parse_args()
    os.environ["MPC_SOURCE"] = a.pool
    n_pool = len(np.load(a.pool, allow_pickle=True)["qpos"])
    rng = np.random.default_rng(a.seed)
    half = n_pool // 2
    idx = np.concatenate([rng.choice(half, a.n // 2, replace=False), half + rng.choice(half, a.n - a.n // 2, replace=False)])
    rng.shuffle(idx)
    t0 = time.time()
    O, A, S, caught_n, done = [], [], [], 0, 0
    with Pool(a.workers, initializer=mh.init_worker, initargs=(a.horizon, a.samples)) as p:
        for r in p.imap_unordered(mh.mpc_state, [int(i) for i in idx]):
            done += 1
            if r["caught"]:
                caught_n += 1
                O.append(r["obs"]); A.append(r["acts"]); S.append(np.full(len(r["acts"]), r["i"]))
            if done % 25 == 0 or done == len(idx):
                np.savez(a.out, obs=np.concatenate(O), acts=np.concatenate(A), state=np.concatenate(S), n_done=done, n_caught=caught_n)
                el = time.time() - t0
                print(f"[progress] {done}/{len(idx)} states, MPC caught {caught_n} ({caught_n / done:.0%}), {el / 60:.0f} min elapsed, ~{el * (len(idx) - done) / done / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
