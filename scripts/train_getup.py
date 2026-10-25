"""End-to-end PPO fine-tune of the single get-up network distilled from the rule-based controller (distill_controller.py), on GetUpEnv (docs 3.24 addendum 13).

1. Value warm-up (the distillation does not train the value head): a few rollouts with the student, lambda = 1 returns (bootstrapped at the 20 s time limit by the critic itself), ONLY the value head is updated.
2. PPO at a low learning rate (the stand-policy recipe: lr 3e-5, target_kl) with the 20 s standing reward. Autosave every 25,000 steps: checkpoints/autosave/ppo_getup_<out>_<steps>_steps.zip.
Usage: python train_getup.py --init-from ../checkpoints/getup_student_distilled.zip --out e2e1 --timesteps 40000000"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CK = os.path.join(HERE, "..", "checkpoints") + "/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", default=CK + "getup_student_distilled.zip")
    ap.add_argument("--out", default="e2e1")
    ap.add_argument("--timesteps", type=int, default=40_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--clip-range", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--log-std", type=float, default=None, help="reset the (state independent) action log-std, e.g. -1.2 (std 0.3)")
    ap.add_argument("--value-warmup", type=int, default=12, help="rollouts (n_envs * n_steps each) used to fit the value head before PPO")
    ap.add_argument("--save-freq", type=int, default=25_000)
    a = ap.parse_args()
    for k in list(os.environ):
        if k.startswith("STANDAI_"):
            del os.environ[k]
    import torch
    torch.set_num_threads(4)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from getup_env import GetUpEnv

    venv = SubprocVecEnv([lambda: GetUpEnv() for _ in range(a.n_envs)], start_method="fork")
    venv.seed(12345)
    custom = dict(learning_rate=a.learning_rate, n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=a.n_epochs, gamma=a.gamma, gae_lambda=0.95, clip_range=a.clip_range,
                  ent_coef=0.0, target_kl=a.target_kl, vf_coef=0.5, max_grad_norm=0.5, verbose=1)
    model = PPO.load(a.init_from, env=venv, device="cpu", custom_objects=custom)
    model.verbose = 1
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(None, ["stdout"]))
    if a.log_std is not None:
        with torch.no_grad():
            model.policy.log_std.data.fill_(a.log_std)
    print(f"[getup] init {a.init_from}: {sum(p.numel() for p in model.policy.parameters()):,} params, action std {torch.exp(model.policy.log_std).mean().item():.2f}", flush=True)

    if a.value_warmup > 0:
        pol = model.policy
        vparams = list(pol.mlp_extractor.value_net.parameters()) + list(pol.value_net.parameters())
        vopt = torch.optim.Adam(vparams, lr=3e-4)
        _, cb = model._setup_learn(10 ** 9, callback=None)
        cb.on_training_start(locals(), globals())
        lam = model.gae_lambda
        model.gae_lambda = 1.0
        model.rollout_buffer.gae_lambda = 1.0
        for it in range(a.value_warmup):
            model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=model.n_steps)
            tot = n = 0.0
            for ep in range(10):
                for rd in model.rollout_buffer.get(a.batch_size):
                    v = pol.predict_values(rd.observations)[:, 0]
                    loss = torch.nn.functional.mse_loss(v, rd.returns)
                    vopt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(vparams, 0.5); vopt.step()
                    tot += float(loss.detach()); n += 1
            print(f"[getup] value warm-up {it + 1}/{a.value_warmup}: value MSE {tot / n:.2f}, mean return {float(model.rollout_buffer.returns.mean()):.1f}", flush=True)
        model.gae_lambda = lam
        model.rollout_buffer.gae_lambda = lam

    os.makedirs(CK + "autosave", exist_ok=True)
    cbk = CheckpointCallback(save_freq=max(a.save_freq // a.n_envs, 1), save_path=CK + "autosave", name_prefix=f"ppo_getup_{a.out}")
    model.learn(total_timesteps=a.timesteps, callback=cbk, reset_num_timesteps=True)
    model.save(CK + f"getup_{a.out}_final.zip")


if __name__ == "__main__":
    main()
