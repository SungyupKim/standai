"""Trains the floor-only get-up policy (floor_env.FloorEnv) from scratch with plain PPO.

Pure RL: exploration help is the same env-level bursts / correlated noise used for the stand policy
(train_locomotion.ColoredNoiseCallback), reward is floor_env's head record + success bonus. Hand-off states
(episodes that reach head >= 0.9 m with a foot down) are counted; the stand policy is a separate lineage.

Usage: train_floor.py [--timesteps N] [--n-envs N] [--out NAME] [--init-from CKPT]
"""
import argparse
import os
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


class AtomicCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that writes to a temp file and renames it: a power cut mid-save can no longer leave a truncated
    .zip behind (two 0-byte v8 checkpoints did). The temp name does not match the '*_steps.zip' globs used elsewhere."""

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            path = self._checkpoint_path(extension="zip")
            tmp = path[:-len(".zip")] + ".tmp.zip"
            self.model.save(tmp)
            os.replace(tmp, path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {path}")
        return True


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


class LevelCurriculumCallback(BaseCallback):
    """Tightens FloorEnv's stand-ready target (level 0 loose -> 1 strict) whenever the recent training success rate clears `gate`.
    The window is cleared after every change, so each level needs `min_episodes` fresh episodes. Level is logged as
    curriculum/level and mirrored to <run>_level.txt (pass it back as --level-start after a restart)."""

    def __init__(self, level, gate=0.4, step=0.05, window=200, min_episodes=100, level_file=None):
        super().__init__()
        self.level, self.gate, self.step, self.min_episodes = float(level), gate, step, min_episodes
        self._succ = deque(maxlen=window)
        self._level_file = level_file

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "floor_success" in info:
                self._succ.append(float(info["floor_success"]))
        if self.level < 1.0 and len(self._succ) >= self.min_episodes and float(np.mean(self._succ)) >= self.gate:
            self.level = min(1.0, self.level + self.step)
            self.training_env.env_method("set_level", self.level)
            self._succ.clear()
            print(f"[floor] curriculum level -> {self.level:.2f} at {self.num_timesteps} steps", flush=True)
            if self._level_file:
                with open(self._level_file, "w") as f:
                    f.write(f"{self.level:.4f}\n")
        self.logger.record("curriculum/level", self.level)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=10_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="ppo_floor_v1")
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--settled", action="store_true", help="v5 experiment: hand-off also needs |vz| < 0.5 m/s and |pitch rate| < 100 deg/s")
    p.add_argument("--standready", action="store_true", help="v6: hand over only in the stand policy's competence region (upright, knees straight, slow); adds crouched starts")
    p.add_argument("--crouch-frac", type=float, default=0.4)
    p.add_argument("--speed-pen", type=float, default=0.0, help="v10: per-step penalty = this x height weight x (|vz|/1 + |pitch rate|/100); 0 = off")
    p.add_argument("--speed-vz-w", type=float, default=1.0, help="weight of |vz| in the speed term (0.25 = rotation speed counts 4x)")
    p.add_argument("--air-pen", type=float, default=0.0, help="v10: per-step penalty when torso > 0.6 m and no foot is on the floor; 0 = off")
    p.add_argument("--max-steps", type=int, default=None, help="episode length in steps (default 300 with --standready, 250 otherwise)")
    p.add_argument("--prone-frac", type=float, default=0.0, help="share of lying starts forced to the start-pitch <= 0 side (inferred prone); natural ~0.2")
    p.add_argument("--level-start", type=float, default=1.0,
                   help="stand-ready target level at launch: 0 = loose ... 1 = strict (default, no curriculum). < 1 enables the level curriculum")
    p.add_argument("--level-gate", type=float, default=0.4, help="training success rate that tightens the level by --level-step")
    p.add_argument("--level-step", type=float, default=0.05)
    p.add_argument("--com-max", type=float, default=None, help="stand-ready also needs COM x - front foot x <= this (m); e.g. 0.06; default off")
    p.add_argument("--no-down-timeout", action="store_true", help="disable env.py's 5 s 'stayed down' KO so only --max-steps ends an episode")
    p.add_argument("--supine-frac", type=float, default=0.0, help="share of lying starts forced to the start-pitch > 0 side (inferred supine); 1.0 = supine only")
    p.add_argument("--learning-rate", type=float, default=None, help="override the (saved) PPO learning rate, e.g. 1e-4 when approx_kl/clip_fraction stay far above ~0.05/0.3")
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

    vec_env = make_vec_env(FloorEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
                           env_kwargs={"settled": args.settled, "standready": args.standready, "crouch_frac": args.crouch_frac,
                                       "speed_pen": args.speed_pen, "air_pen": args.air_pen,
                                       "speed_vz_w": args.speed_vz_w, "max_steps": args.max_steps,
                                       "prone_frac": args.prone_frac, "supine_frac": args.supine_frac,
                                       "down_timeout": not args.no_down_timeout,
                                       "level": args.level_start, "com_max": args.com_max})
    if args.init_from:
        custom = {"ent_coef": args.ent_coef}
        if args.learning_rate is not None:
            custom["learning_rate"] = args.learning_rate
        model = PPO.load(args.init_from, env=vec_env, device=args.device, custom_objects=custom)
        print(f"[floor] warm-starting from {args.init_from}", flush=True)
    else:
        kw = {} if args.learning_rate is None else {"learning_rate": args.learning_rate}
        model = PPO("MlpPolicy", vec_env, device=args.device, verbose=1, ent_coef=args.ent_coef, **kw)
        print("[floor] training from scratch", flush=True)

    tl.LOG_STD_MIN = float(np.log(args.std_min))
    with torch.no_grad():
        model.policy.log_std.data.clamp_(min=tl.LOG_STD_MIN, max=tl.LOG_STD_MAX)

    callbacks = [AtomicCheckpointCallback(save_freq=max(args.save_freq // args.n_envs, 1),
                                     save_path=str(ckpt_dir), name_prefix=run_name),
                 tl.BreakdownCallback(), FloorStatsCallback(), tl.LogStdClampCallback()]
    if args.level_start < 1.0:
        callbacks.append(LevelCurriculumCallback(args.level_start, gate=args.level_gate, step=args.level_step,
                                                 level_file=str(tl.MODELS_DIR / f"{run_name}_level.txt")))
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
