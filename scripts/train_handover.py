"""Trains the hand-over (catch) policy (handover_env.HandoverEnv) from scratch with PPO, every episode starting from a hand-off state of the
pool (prone half by default), offered easy -> hard by how far the COM is ahead of the front foot (train_locomotion.ComLadderPoolCallback),
left/right mirrored copies included.

Usage: train_handover.py [--timesteps N] [--n-envs N] [--out NAME] [--pool ../checkpoints/pool_big_all.npz] [--first-index 16000]
"""
import argparse
import time
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import train_locomotion as tl
from train_floor import AtomicCheckpointCallback
from handover_env import HandoverEnv


class HandoverLadder(tl.ComLadderPoolCallback):
    """Same ladder; the boundary probe is 'the hand-over policy reaches the success state from pool state i without collapsing'."""

    def _hold_from_pool_state(self, i):
        import mujoco
        if self._pool_env is None:
            self._pool_env = HandoverEnv()
        env = self._pool_env
        env.reset(seed=i)
        q, v, _ = self._pool[i]
        env.data.qpos[:], env.data.qvel[:] = q, v
        mujoco.mj_forward(env.model, env.data)
        env._prev_com_x = None
        env._ho_hold = 0
        env._ho_prev_phi = None
        obs = env._obs()
        for _ in range(300):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                return bool(info.get("ho_success"))
        return False


class HandoverStatsCallback(BaseCallback):
    def __init__(self, window=300):
        super().__init__()
        self._succ, self._fail = deque(maxlen=window), deque(maxlen=window)

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "ho_success" in info:
                self._succ.append(float(info["ho_success"]))
                self._fail.append(float(info["ho_fail"]))
        if self._succ:
            self.logger.record("handover/success_rate", float(np.mean(self._succ)))
            self.logger.record("handover/fail_rate", float(np.mean(self._fail)))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="ppo_handover_h1")
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--pool", type=str, default="../checkpoints/pool_big_all.npz")
    p.add_argument("--first-index", type=int, default=16000, help="16000 = prone half of pool_big_all.npz; 0 = both")
    p.add_argument("--last-index", type=int, default=None, help="16000 with --first-index 0 = supine half of pool_big_all.npz")
    p.add_argument("--ladder-start", type=float, default=0.3)
    p.add_argument("--ladder-gate", type=float, default=0.5)
    p.add_argument("--ladder-every", type=int, default=300_000)
    p.add_argument("--ladder-step", type=float, default=0.1)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--ent-coef", type=float, default=0.002)
    p.add_argument("--std-min", type=float, default=0.3)
    p.add_argument("--save-freq", type=int, default=25_000)
    args = p.parse_args()

    tl.MODELS_DIR.mkdir(exist_ok=True)
    ckpt_dir = tl.MODELS_DIR / "autosave"
    ckpt_dir.mkdir(exist_ok=True)
    run_name = f"{args.out}_{time.strftime('%Y%m%d_%H%M%S')}"

    vec_env = make_vec_env(HandoverEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    kw = {"target_kl": args.target_kl}
    if args.learning_rate is not None:
        kw["learning_rate"] = args.learning_rate
    if args.init_from:
        model = PPO.load(args.init_from, env=vec_env, device=args.device,
                         custom_objects={"ent_coef": args.ent_coef, **kw})
        print(f"[handover] warm-starting from {args.init_from}", flush=True)
    else:
        model = PPO("MlpPolicy", vec_env, device=args.device, verbose=1, ent_coef=args.ent_coef, **kw)
        print("[handover] training from scratch", flush=True)
    tl.LOG_STD_MIN = float(np.log(args.std_min))
    with torch.no_grad():
        model.policy.log_std.data.clamp_(min=tl.LOG_STD_MIN, max=tl.LOG_STD_MAX)

    callbacks = [AtomicCheckpointCallback(save_freq=max(args.save_freq // args.n_envs, 1), save_path=str(ckpt_dir), name_prefix=run_name),
                 tl.BreakdownCallback(), HandoverStatsCallback(), tl.LogStdClampCallback(),
                 HandoverLadder(args.pool, frac=1.0, first_index=args.first_index, last_index=args.last_index, grounded=True, start_frac=args.ladder_start, gate=args.ladder_gate,
                                every=args.ladder_every, step=args.ladder_step, mirror=not args.no_mirror)]
    try:
        model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks), reset_num_timesteps=(args.init_from is None))
    finally:
        out = tl.MODELS_DIR / run_name
        model.save(str(out))
        print(f"saved model to {out}.zip", flush=True)


if __name__ == "__main__":
    main()
