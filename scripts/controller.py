"""Rule-based controller that combines the three policies (supine floor policy, prone floor policy, stand policy).

Rules (all parameters are plain constructor arguments so a measurement can sweep them):
  1. SELECT   while the body is low the floor policy is chosen by the sign of the torso pitch: pitch > 0 -> supine policy, <= 0 -> prone policy
              (this is the observable start-side split; "prone" = face down is inferred, see docs 3.17).
  2. HAND OVER the first step the environment reports the stand-ready region (FloorEnv info["floor_success"], whose strictness is the env's `level`)
              -> the stand policy acts from then on.
  3. FALL BACK (optional) if the torso drops below FALL_HEIGHT while the stand policy is in control, the floor policy is re-selected by pitch sign.
              (Measured 2026-10-15: gives no gain over never going back - 24% vs 26% standing at 20 s - but no harm either.)

Usage:
    env = FloorEnv(standready=True, level=1.0, prone_frac=0.0, down_timeout=False, max_steps=10**6)
    ctl = GetUpController(supine_ckpt, prone_ckpt, stand_ckpt)
    obs, _ = env.reset(seed=0); ctl.reset(env)
    for _ in range(1000):
        obs, _, term, trunc, info = env.step(ctl.act(obs, env))
        ctl.observe(env, info)
"""
import numpy as np
from stable_baselines3 import PPO

from env import FALL_HEIGHT


class GetUpController:
    def __init__(self, supine_ckpt, prone_ckpt, stand_ckpt, fall_back=True):
        self.supine = PPO.load(supine_ckpt, device="cpu")
        self.prone = PPO.load(prone_ckpt, device="cpu")
        self.stand = PPO.load(stand_ckpt, device="cpu")
        self.fall_back = fall_back
        self.mode = "floor"
        self.floor = self.supine
        self.floor_name = "supine"
        self.handovers = 0
        self.falls = 0

    @staticmethod
    def pitch(env):
        return float((env.data.qpos[env.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi)

    def _select_floor(self, env):
        if self.pitch(env) <= 0.0:
            self.floor, self.floor_name = self.prone, "prone"
        else:
            self.floor, self.floor_name = self.supine, "supine"

    def reset(self, env):
        self.mode, self.handovers, self.falls = "floor", 0, 0
        self._select_floor(env)

    def act(self, obs, env=None):
        policy = self.stand if self.mode == "stand" else self.floor
        action, _ = policy.predict(obs, deterministic=True)
        return action

    def observe(self, env, info):
        """Call after env.step with the info it returned; updates the mode."""
        z = float(env.data.xpos[env.a_torso_id][2])
        if self.mode == "floor":
            if info.get("floor_success"):
                self.mode = "stand"
                self.handovers += 1
            elif z < FALL_HEIGHT:
                self._select_floor(env)        # re-pick by the current pitch sign while the body is down
        elif z < FALL_HEIGHT:
            self.falls += 1
            if self.fall_back:
                self.mode = "floor"
                self._select_floor(env)
