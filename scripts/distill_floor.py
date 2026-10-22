"""Distils a FLOOR policy (default the supine v12, 64x64) into a larger MLP (default 256x256): behaviour cloning of the mean action (Huber loss: the floor policy's raw means are far outside [-1, 1], variance ~100, because its actions are saturated) + value regression, DAgger style
(same method as distill_stand.py, which worked for the stand policy; docs 3.24 addendum 6). Floor policies trained from scratch as 256x256 nets all failed (loose-target
shortcut, stage-2 stall); this starts the big net from the working policy's behaviour, then PPO continues (train_floor.py --init-from <the saved zip>).

Env = the v12 training/eval configuration (stand-ready region, level 0.7, supine starts only, 450 steps, speed penalty 0.08/0.25). All STANDAI_* training variables other than
STANDAI_NO_RECOVERY_GATE are removed in the workers so nothing leaks in.
Usage: python distill_floor.py [--teacher CKPT] [--supine/--prone side] [--net-arch 256 256] [--out ../checkpoints/floor_distilled_256.zip]"""
import argparse
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CK = os.path.join(HERE, "..", "checkpoints") + "/"
V12 = CK + "autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip"


def clean_env():
    for k in list(os.environ):
        if k.startswith("STANDAI_") and k != "STANDAI_NO_RECOVERY_GATE":
            del os.environ[k]
    os.environ["STANDAI_NO_RECOVERY_GATE"] = "1"


def make_env(side, level):
    from floor_env import FloorEnv
    kw = dict(standready=True, crouch_frac=0.0, max_steps=450, down_timeout=False, level=level, speed_pen=0.08, speed_vz_w=0.25)
    return FloorEnv(prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0, **kw)


def mean_value(model, obs_t):
    import torch
    with torch.no_grad():
        mean = model.policy.get_distribution(obs_t).distribution.mean
        return mean, model.policy.predict_values(obs_t)[:, 0]


def collect(job):
    teacher_path, student_path, n_steps, seed, side, level = job
    clean_env()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    env = make_env(side, level)
    teacher = PPO.load(teacher_path, device="cpu")
    driver = PPO.load(student_path, device="cpu") if student_path else teacher
    obs, _ = env.reset(seed=seed)
    O, A, V = [], [], []
    for _ in range(n_steps):
        m, v = mean_value(teacher, torch.as_tensor(obs[None], dtype=torch.float32))
        O.append(obs.copy()); A.append(m[0].numpy().copy()); V.append(float(v[0]))
        act, _ = driver.predict(obs, deterministic=False)
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            obs, _ = env.reset()
    return np.array(O, dtype=np.float32), np.array(A, dtype=np.float32), np.array(V, dtype=np.float32)


def reach_rate(job):
    path, side, level, n, seed0 = job
    clean_env()
    from stable_baselines3 import PPO
    m = PPO.load(path, device="cpu")
    env = make_env(side, level)
    ok = 0
    for s in range(seed0, seed0 + n):
        obs, _ = env.reset(seed=s)
        for _ in range(450):
            a, _ = m.predict(obs, deterministic=True)
            obs, _, te, tr, info = env.step(a)
            if info.get("floor_success"):
                ok += 1
                break
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=V12)
    ap.add_argument("--side", default="supine", choices=["supine", "prone"])
    ap.add_argument("--level", type=float, default=0.7)
    ap.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--out", default=CK + "floor_distilled_256.zip")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps-round0", type=int, default=250_000, help="per worker")
    ap.add_argument("--steps-later", type=int, default=125_000, help="per worker")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-n", type=int, default=200)
    a = ap.parse_args()
    clean_env()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    teacher = PPO.load(a.teacher, device="cpu")
    venv = DummyVecEnv([lambda: make_env(a.side, a.level)])
    student = PPO("MlpPolicy", venv, device=a.device, verbose=0, policy_kwargs={"net_arch": {"pi": list(a.net_arch), "vf": list(a.net_arch)}},
                  learning_rate=1e-4, n_steps=2048, batch_size=256, n_epochs=10, ent_coef=teacher.ent_coef, target_kl=0.03)
    with torch.no_grad():
        student.policy.log_std.data.copy_(teacher.policy.log_std.data.to(a.device))
    pol = student.policy
    opt = torch.optim.Adam(list(pol.mlp_extractor.parameters()) + list(pol.action_net.parameters()) + list(pol.value_net.parameters()), lr=1e-3)
    tmp = CK + "_distill_floor_student_tmp.zip"
    print(f"teacher {a.teacher} ({sum(x.numel() for x in teacher.policy.parameters()):,} params) -> student {a.net_arch} ({sum(x.numel() for x in pol.parameters()):,} params), {a.side} floor env, level {a.level}", flush=True)

    X = A = V = None
    vscale = None
    with get_context("fork").Pool(a.workers) as pool:
        for r in range(a.rounds):
            t0 = time.time()
            if r == 0:
                parts = pool.map(collect, [(a.teacher, None, a.steps_round0, 1000 + i, a.side, a.level) for i in range(a.workers)])
            else:
                student.save(tmp)
                parts = pool.map(collect, [(a.teacher, tmp, a.steps_later, 5000 + 100 * r + i, a.side, a.level) for i in range(a.workers)])
            o, ac, v = [np.concatenate(x) for x in zip(*parts)]
            X = o if X is None else np.concatenate([X, o]); A = ac if A is None else np.concatenate([A, ac]); V = v if V is None else np.concatenate([V, v])
            if vscale is None:
                vscale = max(float(V.std()), 1.0)
            n = len(X); idx = np.random.default_rng(r).permutation(n); n_val = n // 10
            val, tr = idx[:n_val], idx[n_val:]
            Xt, At, Vt = (torch.as_tensor(z, device=a.device) for z in (X, A, V))
            print(f"round {r}: {len(o):,} new samples (total {n:,}, {time.time() - t0:.0f}s); teacher value mean {V.mean():.1f} std {V.std():.1f}", flush=True)
            for ep in range(a.epochs):
                perm = tr[np.random.default_rng(1000 * r + ep).permutation(len(tr))]
                for g in opt.param_groups:
                    g["lr"] = 1e-3 * (1.0 - 0.9 * ep / a.epochs)
                tot = 0.0
                for i in range(0, len(perm), 4096):
                    b = torch.as_tensor(perm[i:i + 4096], device=a.device)
                    lp, lv = pol.mlp_extractor(Xt[b])
                    loss = torch.nn.functional.smooth_l1_loss(pol.action_net(lp), At[b], beta=1.0) + a.value_weight * torch.nn.functional.huber_loss(pol.value_net(lv)[:, 0] / vscale, Vt[b] / vscale)
                    opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(b)
                if ep % 20 == 19 or ep == a.epochs - 1:
                    with torch.no_grad():
                        b = torch.as_tensor(val, device=a.device)
                        lp, lv = pol.mlp_extractor(Xt[b])
                        a_mse = float(torch.nn.functional.smooth_l1_loss(pol.action_net(lp), At[b], beta=1.0)); v_mae = float((pol.value_net(lv)[:, 0] - Vt[b]).abs().mean())
                    print(f"   epoch {ep + 1:3d}: train loss {tot / len(perm):.4f} | held-out action Huber {a_mse:.4f} (action variance {float(At[b].var()):.3f}) | value MAE {v_mae:.2f}", flush=True)
        student.save(a.out)
        print(f"saved {a.out}", flush=True)
        # the project's own measurement: the hand-off region reached within 450 steps, deterministic, fixed seeds, teacher vs student
        for name, path in (("teacher", a.teacher), ("student", a.out)):
            ks = pool.map(reach_rate, [(path, a.side, a.level, a.eval_n // a.workers, 25000 + i * (a.eval_n // a.workers)) for i in range(a.workers)])
            print(f"{name:8s}: reached the level-{a.level} region in {sum(ks)}/{(a.eval_n // a.workers) * a.workers} starts ({sum(ks) / ((a.eval_n // a.workers) * a.workers):.0%})", flush=True)


if __name__ == "__main__":
    main()
