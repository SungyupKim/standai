"""Rule-based controller that combines the three policies (supine floor policy, prone floor policy, stand policy).

Rules (all parameters are plain constructor arguments so a measurement can sweep them):
  1. SELECT   while the body is low the floor policy is chosen by the sign of the torso pitch: pitch > 0 -> supine policy, <= 0 -> prone policy
              (this is the observable start-side split; "prone" = face down is inferred, see docs 3.17).
  2. HAND OVER the first step the environment reports the stand-ready region (FloorEnv info["floor_success"], whose strictness is the env's `level`)
              -> the stand policy acts from then on.
  2b. GATE    (optional, com_gate=<m>) for two-footed hand-overs only (feet height gap <= 0.3 m): if the centre of mass is more than `com_gate` metres ahead
              of the FRONT foot, keep the floor policy for up to `wait_steps` more steps and hand over as soon as it is behind the gate (or when the wait is
              over). The stand policy catches 84% of prone hand-offs with COM - front foot < -0.05 m and 0% above +0.15 m (docs 3.22). One-leg-raised
              hand-overs (v12) are not gated: the relation is reversed there.
  2c. HAND-OVER POLICY (optional, handover_ckpt=...) floor -> hand-over policy (scripts/handover_env.py: moves the body into a state the stand policy holds) -> stand
              when its success condition has held for HOLD_STEPS steps (falls back to the floor policy if the torso drops below FAIL_TORSO, time-out -> stand).
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
from sym import load_stand


class GetUpController:
    def __init__(self, supine_ckpt, prone_ckpt, stand_ckpt, fall_back=True, com_gate=None, wait_steps=25, handover_ckpt=None, handover_max=300, handover_sides=("prone",)):
        self.supine = PPO.load(supine_ckpt, device="cpu")
        self.prone = PPO.load(prone_ckpt, device="cpu")
        self.stand = load_stand(stand_ckpt)      # STANDAI_SYM_STAND=1 -> left-right symmetrised (scripts/sym.py)
        self.fall_back = fall_back
        self.handover = PPO.load(handover_ckpt, device="cpu") if handover_ckpt else None   # optional hand-over policy between floor and stand
        self.handover_max = handover_max
        self.handover_sides = tuple(handover_sides)     # which floor policy's arrivals go through the hand-over policy (it was trained on prone hand-offs)
        self._ho_steps = 0
        self._ho_hold = 0
        self.ho_success = 0
        self.ho_fail = 0
        self.mode = "floor"
        self.floor = self.supine
        self.floor_name = "supine"
        self.handovers = 0
        self.falls = 0
        self.com_gate, self.wait_steps = com_gate, wait_steps
        self._waited = 0
        self.forced = 0

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
        self._waited, self.forced = 0, 0
        self._ho_steps = self._ho_hold = self.ho_success = self.ho_fail = 0
        self._select_floor(env)

    @staticmethod
    def com_ahead_of_front_foot(env):
        """(COM x - front foot x, feet height gap) in metres."""
        d = env.data
        com_x = float(np.sum(env._a_masses * d.xipos[env._a_bodies, 0]) / env._a_masses.sum())
        fx = [float(d.site_xpos[s][0]) for s in env.a_feet]
        fz = [float(d.site_xpos[s][2]) for s in env.a_feet]
        return com_x - max(fx), abs(fz[0] - fz[1])

    def _may_hand_over(self, env):
        if self.com_gate is None:
            return True
        ahead, gap = self.com_ahead_of_front_foot(env)
        if gap > 0.3 or ahead <= self.com_gate:
            return True
        self._waited += 1
        if self._waited >= self.wait_steps:
            self.forced += 1
            return True
        return False

    def act(self, obs, env=None):
        policy = self.stand if self.mode == "stand" else (self.handover if self.mode == "handover" else self.floor)
        action, _ = policy.predict(obs, deterministic=True)
        return action

    def observe(self, env, info):
        """Call after env.step with the info it returned; updates the mode."""
        z = float(env.data.xpos[env.a_torso_id][2])
        if self.mode == "floor":
            if info.get("floor_success") and self._may_hand_over(env):
                self.mode = "handover" if (self.handover is not None and self.floor_name in self.handover_sides) else "stand"
                self.handovers += 1
                self._waited = 0
                self._ho_steps = self._ho_hold = 0
            elif z < FALL_HEIGHT:
                self._select_floor(env)        # re-pick by the current pitch sign while the body is down
        elif self.mode == "handover":
            from handover_env import FAIL_TORSO, HOLD_STEPS, ok_state
            self._ho_steps += 1
            self._ho_hold = self._ho_hold + 1 if ok_state(env) else 0
            if self._ho_hold >= HOLD_STEPS:
                self.mode = "stand"; self.ho_success += 1
            elif z < FAIL_TORSO:
                self.mode = "floor"; self.ho_fail += 1; self._select_floor(env)
            elif self._ho_steps >= self.handover_max:
                self.mode = "stand"          # time out: let the stand policy try anyway
        elif z < FALL_HEIGHT:
            self.falls += 1
            if self.fall_back:
                self.mode = "floor"
                self._select_floor(env)
