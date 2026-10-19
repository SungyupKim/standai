"""Deterministic hand-off catch (benchmark criterion) per COM-ahead bin, several stand checkpoints side by side, on one or more eval sets.
Usage: STANDAI_NO_RECOVERY_GATE=1 .venv/bin/python -I scripts/analysis/catch_by_com.py CKPT [CKPT ...] [--sets eval2_all eval_strict_all]"""
import argparse, os, sys
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
import numpy as np, mujoco
from multiprocessing import Pool

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--sets", nargs="+", default=["eval2_all", "eval_strict_all"])
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
BINS = [(-9, -0.05), (-0.05, 0), (0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 9)]
_G = {}

def job(args):
    ck, setname, i = args
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from locomotion_env import LocomotionEnv
    if ck not in _G: _G[ck] = PPO.load(ck, device="cpu")
    if "env" not in _G:
        _G["env"] = LocomotionEnv(stand_only=True); _G["env"].set_calm_frac(1.0)
    d = np.load(ROOT + f"checkpoints/{setname}.npz", allow_pickle=True)
    env, model = _G["env"], _G[ck]
    env.reset(seed=i)
    env.data.qpos[:], env.data.qvel[:] = d["qpos"][i], d["qvel"][i]
    mujoco.mj_forward(env.model, env.data); env._prev_com_x = None
    obs = env._obs()
    com = np.sum(env._a_masses * env.data.xipos[env._a_bodies, 0]) / env._a_masses.sum()
    ahead = com - max(env.data.site_xpos[s][0] for s in env.a_feet)
    run = 0
    for _ in range(150):
        act, _ = model.predict(obs, deterministic=True)
        obs, *_ = env.step(act)
        ok = (env.data.xpos[env.a_torso_id][2] >= 0.7 and env.data.xpos[env.a_head_id][2] >= 0.9
              and abs(wrap(env.data.qpos[env.a_ry_qpos])) < np.radians(40))
        run = run + 1 if ok else 0
    return ahead, run >= 20

if __name__ == "__main__":
    with Pool(a.workers) as p:
        for s in a.sets:
            n = len(np.load(ROOT + f"checkpoints/{s}.npz", allow_pickle=True)["qpos"])
            print(f"== {s} (first {n // 2} supine, last {n - n // 2} prone) ==")
            res = {}
            for ck in a.ckpts:
                r = np.array(p.map(job, [(ck, s, i) for i in range(n)]), dtype=float)
                res[ck] = r
            names = [os.path.basename(c)[:34] for c in a.ckpts]
            print("side/bin            n  " + "  ".join(f"{x[-22:]:>22}" for x in names))
            for side, sl in (("supine", slice(0, n // 2)), ("prone", slice(n // 2, n))):
                print(f"{side}: all        {n//2:4d}  " + "  ".join(f"{res[c][sl,1].mean():>21.0%} " for c in a.ckpts))
                if side == "prone":
                    for lo, hi in BINS:
                        m = (res[a.ckpts[0]][sl, 0] >= lo) & (res[a.ckpts[0]][sl, 0] < hi)
                        if m.any():
                            print(f"  {lo:+.2f}..{hi:+.2f}  {m.sum():4d}  " + "  ".join(f"{res[c][sl][m,1].mean():>21.0%} " for c in a.ckpts))
