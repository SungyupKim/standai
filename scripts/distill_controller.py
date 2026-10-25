"""Distils the WHOLE rule-based controller (GetUpController: supine / prone floor policies, prone hand-over policy, stand policy + its switching rules) into ONE network (docs 3.24 addendum 13).

DAgger: the student drives (mixed with the teacher with probability beta, small action noise), the teacher controller runs on the same environment (GetUpEnv, 20 s episodes, random side) and labels every
visited state with its own action; behaviour cloning (MSE on the teacher's clipped mean action) on the aggregated data; after every round the student is measured on fixed seeds (standing at 10 s / 20 s,
the controller benchmark's criterion) next to the teacher. The saved zip is an SB3 PPO whose value head is NOT trained here (train_getup.py fits it before the PPO fine-tune).
Usage: python distill_controller.py [--net-arch 512 512] [--rounds 6] [--out ../checkpoints/getup_student.zip]"""
import argparse
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CK = os.path.join(HERE, "..", "checkpoints") + "/"
CKPTS = dict(supine=CK + "best_floor_supine_fdist.zip", prone=CK + "best_prone8.zip", stand=CK + "best_stand_big1.zip", handover=CK + "best_handover_prone.zip")


def clean_env():
    for k in list(os.environ):
        if k.startswith("STANDAI_"):
            del os.environ[k]


def make_ctl(handover_side=("prone",)):
    from controller import GetUpController
    return GetUpController(CKPTS["supine"], CKPTS["prone"], CKPTS["stand"], fall_back=True, handover_ckpt=CKPTS["handover"], handover_sides=handover_side)


def collect(job):
    student_path, beta, n_steps, seed, noise = job
    clean_env()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from getup_env import GetUpEnv
    env = GetUpEnv()
    ctl = make_ctl()
    student = PPO.load(student_path, device="cpu") if student_path else None
    rng = np.random.default_rng(seed)
    O, A, M = [], [], []
    obs, _ = env.reset(seed=seed); ctl.reset(env)
    for _ in range(n_steps):
        a_t = np.array(ctl.act(obs, env), dtype=np.float32)
        O.append(obs.copy()); A.append(a_t.copy()); M.append({"floor": 0, "handover": 1, "stand": 2}[ctl.mode] + 3 * (ctl.floor_name == "prone"))
        if student is None or rng.random() < beta:
            a_exec = a_t
        else:
            a_exec = student.predict(obs, deterministic=True)[0]
        a_exec = np.clip(a_exec + noise * rng.standard_normal(a_exec.shape), -1.0, 1.0)
        obs, _, term, trunc, info = env.step(a_exec)
        ctl.observe(env, info)
        if trunc:
            obs, _ = env.reset(); ctl.reset(env)
    return np.array(O, dtype=np.float32), np.array(A, dtype=np.float32), np.array(M, dtype=np.int8)


def evaluate(job):
    """Standing at 10 s / 20 s per start (the controller benchmark's criterion) for the student zip, or for the rule-based teacher when path is None."""
    path, side, seed0, n = job
    clean_env()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from floor_env import FloorEnv
    from controller import GetUpController
    env = FloorEnv(standready=True, level=0.7, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False, prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    model = PPO.load(path, device="cpu") if path else None
    ctl = make_ctl(("prone",)) if model is None else GetUpController(CKPTS["supine"], CKPTS["prone"], CKPTS["stand"], fall_back=True)
    s10 = send = 0
    for seed in range(seed0, seed0 + n):
        obs, _ = env.reset(seed=seed); ctl.reset(env)
        hist = []
        for _ in range(1000):
            a = ctl.act(obs, env) if model is None else model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(a)
            if model is None:
                ctl.observe(env, info)
            hist.append((float(env.data.xpos[env.a_torso_id][2]), GetUpController.pitch(env)))
        ok = lambda lo, hi: all(z >= 0.95 and abs(np.degrees(p)) < 30 for z, p in hist[lo:hi])
        s10 += ok(475, 500); send += ok(975, 1000)
    return s10, send


def eval_pool(pool, path, n, seed0, workers):
    k = n // workers
    out = {}
    for side in ("supine", "prone"):
        r = pool.map(evaluate, [(path, side, seed0 + i * k, k) for i in range(workers)])
        out[side] = (sum(x[0] for x in r), sum(x[1] for x in r), k * workers)
    return out


def fmt(res):
    s, p = res["supine"], res["prone"]
    return f"supine {s[1]}/{s[2]} ({s[1] / s[2]:.0%}), prone {p[1]}/{p[2]} ({p[1] / p[2]:.0%}), 50:50 {(s[1] / s[2] + p[1] / p[2]) / 2:.1%} (10 s: {(s[0] / s[2] + p[0] / p[2]) / 2:.0%})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-arch", type=int, nargs="+", default=[512, 512])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps-per-worker", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-n", type=int, default=96, help="starts per side in the per-round measurement (multiple of workers)")
    ap.add_argument("--eval-seed0", type=int, default=90000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=CK + "getup_student.zip")
    ap.add_argument("--notify", action="store_true")
    a = ap.parse_args()
    clean_env()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from getup_env import GetUpEnv

    def say(msg):
        print(msg, flush=True)
        if a.notify:
            from kakao_notify import send_message
            send_message(msg)

    student = PPO("MlpPolicy", DummyVecEnv([lambda: GetUpEnv()]), device=a.device, verbose=0, policy_kwargs={"net_arch": {"pi": list(a.net_arch), "vf": list(a.net_arch)}})
    with torch.no_grad():
        student.policy.log_std.data.fill_(-1.2)
    pol = student.policy
    opt = torch.optim.Adam(list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters()), lr=1e-3)
    tmp = CK + "_getup_student_tmp.zip"
    betas = [1.0, 0.5, 0.25] + [0.0] * 20
    X = A = None
    with get_context("fork").Pool(a.workers) as pool:
        base = eval_pool(pool, None, a.eval_n, a.eval_seed0, a.workers)
        say(f"[distill] teacher (rule-based controller) on seeds {a.eval_seed0}+, {a.eval_n}/side: {fmt(base)}")
        for r in range(a.rounds):
            t0 = time.time()
            student.save(tmp)
            parts = pool.map(collect, [(tmp if r else None, betas[r], a.steps_per_worker, 20000 + 100 * r + i, a.noise) for i in range(a.workers)])
            o, ac, m = [np.concatenate(x) for x in zip(*parts)]
            X = o if X is None else np.concatenate([X, o]); A = ac if A is None else np.concatenate([A, ac])
            Xt, At = torch.as_tensor(X, device=a.device), torch.as_tensor(A, device=a.device)
            n = len(X); idx = np.random.default_rng(r).permutation(n); n_val = n // 20
            val, tr = idx[:n_val], idx[n_val:]
            for ep in range(a.epochs):
                perm = tr[np.random.default_rng(1000 * r + ep).permutation(len(tr))]
                for g in opt.param_groups:
                    g["lr"] = 1e-3 * (1.0 - 0.9 * ep / a.epochs)
                for i in range(0, len(perm), 4096):
                    b = torch.as_tensor(perm[i:i + 4096], device=a.device)
                    loss = torch.nn.functional.mse_loss(pol.action_net(pol.mlp_extractor.policy_net(Xt[b])), At[b])
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                b = torch.as_tensor(val, device=a.device)
                vl = float(torch.nn.functional.mse_loss(pol.action_net(pol.mlp_extractor.policy_net(Xt[b])), At[b]))
            student.save(a.out)
            res = eval_pool(pool, a.out, a.eval_n, a.eval_seed0, a.workers)
            modes = np.bincount(m, minlength=6)
            say(f"[distill] round {r} beta {betas[r]}: {len(X):,} samples, held-out action MSE {vl:.4f}, {time.time() - t0:.0f}s; student {fmt(res)} (teacher {fmt(base)}); teacher modes in this round supine floor/handover/stand, prone: {modes.tolist()}")
    say(f"[distill] done, saved {a.out}")


if __name__ == "__main__":
    main()
