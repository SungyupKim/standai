"""Distils a stand-style policy (default best_stand_v55b, 64x64) into a larger MLP (default 256x256) by behaviour cloning of its MEAN action plus value regression (DAgger style).

Why: the three small policies are 18K-parameter nets; a 256x256 floor policy learns much faster and higher on the loose target (docs 3.24 addendum 4). The stand and hand-over policies
cannot be trained from scratch (every attempt failed) and the weights of a 64x64 net cannot be copied into a 256x256 one, so the big net first imitates the small one, then PPO continues
from it (train_locomotion.py --init-from <the saved zip>).

Data: the teacher's observations come from the stand training distribution (calm / harsh starts, hand-off pool replay 40%, hand-off-like velocities, blended starts; the v54/v55b
command) -- round 0 the teacher drives (sampled actions), later rounds the STUDENT drives (sampled) and the teacher labels the visited states (mean action, value).
Loss: MSE(student mean, teacher mean) + HUBER(student value/100, teacher value/100). Evaluates the student with the project's own measurements (calm gate, hand-off catch).
Usage: STANDAI_NO_RECOVERY_GATE=1 python distill_stand.py [--teacher ../checkpoints/best_stand_v55b.zip] [--net-arch 256 256] [--out ../checkpoints/stand_distilled_256.zip]
"""
import argparse
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
POOL = os.path.join(HERE, "..", "checkpoints", "pool_big_all.npz")


def set_env_vars():
    os.environ.setdefault("STANDAI_NO_RECOVERY_GATE", "1")
    os.environ.setdefault("STANDAI_HARSH_MIN_TORSO", "0.7")
    os.environ.setdefault("STANDAI_START_VEL_POOL", POOL)
    os.environ.setdefault("STANDAI_START_VEL_PROB", "0.6")
    os.environ.setdefault("STANDAI_START_VEL_JOINT", "3.5")
    os.environ.setdefault("STANDAI_MIX_POOL", POOL)
    os.environ.setdefault("STANDAI_MIX_PROB", "0.3")
    os.environ.setdefault("STANDAI_MIX_LAM_LO", "0.7")


def raw_mean_and_value(model, obs_t):
    import torch
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t)
        mean = dist.distribution.mean
        value = model.policy.predict_values(obs_t)[:, 0]
    return mean, value


def collect(job):
    """One worker: n steps in the stand environment; actions executed come from the teacher (student_path None) or the student, both sampled; labels always the teacher's."""
    teacher_path, student_path, n_steps, seed = job
    set_env_vars()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from locomotion_env import LocomotionEnv
    env = LocomotionEnv(stand_only=True)
    env.set_calm_frac(0.5)
    d = np.load(POOL, allow_pickle=True)
    env.set_replay_pool([(q, v, False) for q, v in zip(d["qpos"], d["qvel"])], 0.4)
    teacher = PPO.load(teacher_path, device="cpu")
    driver = PPO.load(student_path, device="cpu") if student_path else teacher
    obs, _ = env.reset(seed=seed)
    O, A, V = [], [], []
    for _ in range(n_steps):
        o_t = torch.as_tensor(obs[None], dtype=torch.float32)
        m, v = raw_mean_and_value(teacher, o_t)
        O.append(obs.copy()); A.append(m[0].numpy().copy()); V.append(float(v[0]))
        act, _ = driver.predict(obs, deterministic=False)
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            obs, _ = env.reset()
    return np.array(O, dtype=np.float32), np.array(A, dtype=np.float32), np.array(V, dtype=np.float32)


def gather(pool, teacher_path, student_path, per_worker, workers, seed0):
    parts = pool.map(collect, [(teacher_path, student_path, per_worker, seed0 + i) for i in range(workers)])
    return [np.concatenate(x) for x in zip(*parts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.path.join(HERE, "..", "checkpoints", "best_stand_v55b.zip"))
    ap.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "checkpoints", "stand_distilled_256.zip"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps-round0", type=int, default=200_000, help="per worker")
    ap.add_argument("--steps-later", type=int, default=100_000, help="per worker")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    set_env_vars()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from locomotion_env import LocomotionEnv

    teacher = PPO.load(a.teacher, device="cpu")
    venv = DummyVecEnv([lambda: LocomotionEnv(stand_only=True)])
    student = PPO("MlpPolicy", venv, device=a.device, verbose=0, policy_kwargs={"net_arch": {"pi": list(a.net_arch), "vf": list(a.net_arch)}},
                  learning_rate=5e-5, n_steps=2048, batch_size=256, n_epochs=10, ent_coef=teacher.ent_coef, target_kl=0.03)
    with torch.no_grad():
        student.policy.log_std.data.copy_(teacher.policy.log_std.data.to(a.device))          # same exploration std as the teacher
    pol = student.policy
    params = list(pol.mlp_extractor.parameters()) + list(pol.action_net.parameters()) + list(pol.value_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    tmp_student = os.path.join(HERE, "..", "checkpoints", "_distill_student_tmp.zip")
    print(f"teacher {a.teacher} ({sum(x.numel() for x in teacher.policy.parameters()):,} params) -> student {a.net_arch} ({sum(x.numel() for x in pol.parameters()):,} params)", flush=True)

    X = A = V = None
    ctx = get_context("fork")
    with ctx.Pool(a.workers) as pool:
        for r in range(a.rounds):
            t0 = time.time()
            if r == 0:
                o, ac, v = gather(pool, a.teacher, None, a.steps_round0, a.workers, 1000)
            else:
                student.save(tmp_student)
                o, ac, v = gather(pool, a.teacher, tmp_student, a.steps_later, a.workers, 5000 + 100 * r)
            X = o if X is None else np.concatenate([X, o]); A = ac if A is None else np.concatenate([A, ac]); V = v if V is None else np.concatenate([V, v])
            n = len(X); idx = np.random.default_rng(r).permutation(n); n_val = n // 10
            val, tr = idx[:n_val], idx[n_val:]
            Xt, At, Vt = (torch.as_tensor(z, device=a.device) for z in (X, A, V))
            print(f"round {r}: {len(o):,} new samples (total {n:,}, collected in {time.time() - t0:.0f}s); teacher value mean {V.mean():.0f} std {V.std():.0f}", flush=True)
            for ep in range(a.epochs):
                perm = tr[np.random.default_rng(1000 * r + ep).permutation(len(tr))]
                for g in opt.param_groups:
                    g["lr"] = 1e-3 * (1.0 - 0.9 * ep / a.epochs)
                tot = 0.0
                for i in range(0, len(perm), 4096):
                    b = torch.as_tensor(perm[i:i + 4096], device=a.device)
                    latent_pi, latent_vf = pol.mlp_extractor(Xt[b])
                    mean, val_pred = pol.action_net(latent_pi), pol.value_net(latent_vf)[:, 0]
                    loss = torch.nn.functional.mse_loss(mean, At[b]) + a.value_weight * torch.nn.functional.huber_loss(val_pred / 100.0, Vt[b] / 100.0)
                    opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
                if ep % 10 == 9 or ep == a.epochs - 1:
                    with torch.no_grad():
                        b = torch.as_tensor(val, device=a.device)
                        latent_pi, latent_vf = pol.mlp_extractor(Xt[b])
                        a_mse = float(torch.nn.functional.mse_loss(pol.action_net(latent_pi), At[b]))
                        v_mae = float((pol.value_net(latent_vf)[:, 0] - Vt[b]).abs().mean())
                    print(f"   epoch {ep + 1:3d}: train loss {tot / len(perm):.4f} | held-out action MSE {a_mse:.4f} (teacher action variance {float(At[b].var()):.3f}) | value MAE {v_mae:.1f}", flush=True)
    student.save(a.out)
    print(f"saved {a.out}", flush=True)
    if not a.skip_eval:
        from stand_progress_notify import gate_survival_rate_calm_only, handoff_catch
        eval_set = os.path.join(HERE, "..", "checkpoints", "eval2_all.npz")
        for name, path in (("teacher", a.teacher), ("student", a.out)):
            up, end, n = handoff_catch(path, eval_set)
            print(f"{name:8s}: calm gate {gate_survival_rate_calm_only(path)}  | hand-off catch {up}/{n}  (torso>=0.7 at the end {end})", flush=True)


if __name__ == "__main__":
    main()
