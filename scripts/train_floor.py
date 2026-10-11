"""Trains the floor-only get-up policy (floor_env.FloorEnv) from scratch with plain PPO.

Pure RL: exploration help is the same env-level bursts / correlated noise used for the stand policy
(train_locomotion.ColoredNoiseCallback), reward is floor_env's head record + success bonus. Hand-off states
(episodes that reach head >= 0.9 m with a foot down) are counted; the stand policy is a separate lineage.

Usage: train_floor.py [--timesteps N] [--n-envs N] [--out NAME] [--init-from CKPT]
"""
import argparse
import time
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import train_locomotion as tl
from floor_env import FloorEnv


class FloorStatsCallback(BaseCallback):
    """Success rate (head >= 0.9 m, foot down) and mean peak head height over the last `window` episodes."""

    def __init__(self, window=200):
        super().__init__()
        self._succ = deque(maxlen=window)
        self._head = deque(maxlen=window)
        self._h05 = deque(maxlen=window)

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "floor_success" in info:
                self._succ.append(float(info["floor_success"]))
                self._head.append(info["floor_head_max"])
                self._h05.append(float(info["floor_head_max"] > 0.5))
        if self._succ:
            self.logger.record("floor/success_rate", float(np.mean(self._succ)))
            self.logger.record("floor/head_max_mean", float(np.mean(self._head)))
            self.logger.record("floor/head_over_0.5", float(np.mean(self._h05)))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=10_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="ppo_floor_v1")
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--save-freq", type=int, default=25_000)
    p.add_argument("--ent-coef", type=float, default=0.002)
    p.add_argument("--std-min", type=float, default=0.45)
    p.add_argument("--colored-noise-rho", type=float, default=0.9)
    p.add_argument("--floor-burst-rate", type=float, default=0.1)
    p.add_argument("--floor-burst-hold", type=int, default=10)
    args = p.parse_args()

    tl.MODELS_DIR.mkdir(exist_ok=True)
    ckpt_dir = tl.MODELS_DIR / "autosave"
    ckpt_dir.mkdir(exist_ok=True)
    run_name = f"{args.out}_{time.strftime('%Y%m%d_%H%M%S')}"

    vec_env = make_vec_env(FloorEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    if args.init_from:
        model = PPO.load(args.init_from, env=vec_env, device=args.device,
                         custom_objects={"ent_coef": args.ent_coef})
        print(f"[floor] warm-starting from {args.init_from}", flush=True)
    else:
        model = PPO("MlpPolicy", vec_env, device=args.device, verbose=1, ent_coef=args.ent_coef)
        print("[floor] training from scratch", flush=True)

    tl.LOG_STD_MIN = float(np.log(args.std_min))
    with torch.no_grad():
        model.policy.log_std.data.clamp_(min=tl.LOG_STD_MIN, max=tl.LOG_STD_MAX)

    callbacks = [CheckpointCallback(save_freq=max(args.save_freq // args.n_envs, 1),
                                     save_path=str(ckpt_dir), name_prefix=run_name),
                 tl.BreakdownCallback(), FloorStatsCallback(), tl.LogStdClampCallback()]
    if args.colored_noise_rho > 0.0:
        callbacks.append(tl.ColoredNoiseCallback(rho=args.colored_noise_rho, floor_scale=1.0,
                                                  burst_rate=args.floor_burst_rate,
                                                  burst_hold=args.floor_burst_hold))
    try:
        model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks),
                    reset_num_timesteps=(args.init_from is None))
    finally:
        out = tl.MODELS_DIR / run_name
        model.save(str(out))
        print(f"saved model to {out}.zip", flush=True)


if __name__ == "__main__":
    main()
