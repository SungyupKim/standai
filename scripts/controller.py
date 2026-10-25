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
import os

import numpy as np
from stable_baselines3 import PPO

from env import FALL_HEIGHT
from sym import load_stand


class GetUpController:
    def __init__(self, supine_ckpt, prone_ckpt, stand_ckpt, fall_back=True, com_gate=None, wait_steps=25, handover_ckpt=None, handover_max=300, handover_sides=("prone",), v_gate=None, v_wait=50, floor_flip=0, flip_sides=("supine", "prone"), handover_forever=False, floor_sample=False):
        self.supine = PPO.load(supine_ckpt, device="cpu")
        self.prone = PPO.load(prone_ckpt, device="cpu")
        self.stand = load_stand(stand_ckpt)      # STANDAI_SYM_STAND=1 -> left-right symmetrised (scripts/sym.py)
        self.fall_back = fall_back
        self.handover = PPO.load(handover_ckpt, device="cpu") if handover_ckpt else None   # optional hand-over policy between floor and stand
        self.handover_max = handover_max
        self.handover_sides = tuple(handover_sides)     # which floor policy's arrivals go through the hand-over policy (it was trained on prone hand-offs)
        self.floor_flip, self.flip_sides = int(floor_flip), tuple(flip_sides)   # every floor_flip steps the floor policy is shown the left-right swapped body (docs 3.24 addendum 3)
        self.floor_sample = floor_sample               # sample the floor policy's actions (as in training) instead of using its mean; big networks trained with large exploration noise can depend on it
        self.handover_forever = handover_forever       # the hand-over policy keeps control (never passes to the stand policy); only a collapse sends it back to the floor policy
        self._floor_steps, self._flipped = 0, False
        self.v_gate, self.v_wait = v_gate, v_wait          # hand over only when the hand-over policy's critic V(s) >= v_gate (or after v_wait steps of waiting)
        self._arrived = False
        self._arrive_wait = 0
        self.v_forced = 0
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
        # diagnostic (docs 3.24 addendum 10): wait with the hand-over until the COM moves forward at >= STANDAI_VX_GATE m/s (at most STANDAI_VX_WAIT steps); set per command with `env`, never export
        _g = os.environ.get("STANDAI_VX_GATE")
        self.vx_gate = float(_g) if _g else None
        self.vx_wait = int(os.environ.get("STANDAI_VX_WAIT", "50"))
        self._vx_waited, self._com_prev, self._vx, self.vx_forced = 0, None, 0.0, 0
        # diagnostic (docs 3.24 addendum 12): hand over only when a learned catchability score of the arrival state (checkpoints/arrival_clf.pt, scripts/analysis/train_arrival_clf.py) is >= STANDAI_CLF_GATE,
        # waiting at most STANDAI_CLF_WAIT steps; set per command with `env`, never export
        _c = os.environ.get("STANDAI_CLF_GATE")
        self.clf_gate = float(_c) if _c else None
        self.clf_wait = int(os.environ.get("STANDAI_CLF_WAIT", "50"))
        self._clf_waited, self.clf_forced, self._clf = 0, 0, None
        if self.clf_gate is not None:
            import torch
            ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "arrival_clf.pt"))
            self._clf = []
            for st in ck["states"]:
                n = torch.nn.Sequential(torch.nn.Linear(33, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1)); n.load_state_dict(st); self._clf.append(n)
            self._clf_mu, self._clf_sd = ck["mu"], ck["sd"]

    def catch_score(self, env):
        """Mean over the classifier ensemble of P(stand policy catches this state) for the current body state."""
        import torch
        d = env.data
        x = torch.tensor(np.concatenate([d.qpos[1:17], d.qvel[0:17]]), dtype=torch.float32)[None]
        x = (x - self._clf_mu) / self._clf_sd
        with torch.no_grad():
            return float(np.mean([torch.sigmoid(n(x))[0, 0].item() for n in self._clf]))

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
        self._vx_waited, self._com_prev, self._vx, self.vx_forced = 0, None, 0.0, 0
        self._clf_waited = 0
        self._ho_steps = self._ho_hold = self.ho_success = self.ho_fail = 0
        self._arrived, self._arrive_wait, self.v_forced = False, 0, 0
        self._floor_steps, self._flipped = 0, False
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
        if self.clf_gate is not None and self.catch_score(env) < self.clf_gate:
            self._clf_waited += 1
            if self._clf_waited < self.clf_wait:
                return False
            self.clf_forced += 1
        if self.vx_gate is not None and self._vx < self.vx_gate:
            self._vx_waited += 1
            if self._vx_waited < self.vx_wait:
                return False
            self.vx_forced += 1
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

    def _handover_value(self, obs):
        import torch
        with torch.no_grad():
            return float(self.handover.policy.predict_values(torch.as_tensor(obs[None], dtype=torch.float32))[0, 0])

    def act(self, obs, env=None):
        if self.mode == "floor" and self._arrived:
            self._arrive_wait += 1
            if self._handover_value(obs) >= self.v_gate or self._arrive_wait >= self.v_wait:
                if self._arrive_wait >= self.v_wait:
                    self.v_forced += 1
                self.mode = "handover"
                self.handovers += 1
                self._arrived = False
                self._ho_steps = self._ho_hold = 0
        policy = self.stand if self.mode == "stand" else (self.handover if self.mode == "handover" else self.floor)
        if self.mode == "floor" and self.floor_flip and self.floor_name in self.flip_sides:
            if self._floor_steps > 0 and self._floor_steps % self.floor_flip == 0:
                self._flipped = not self._flipped
            self._floor_steps += 1
            if self._flipped:
                from sym import swap_action, swap_obs
                action, _ = policy.predict(swap_obs(obs), deterministic=True)
                return swap_action(action)
        action, _ = policy.predict(obs, deterministic=not (self.floor_sample and self.mode == "floor"))
        return action

    def observe(self, env, info):
        """Call after env.step with the info it returned; updates the mode."""
        z = float(env.data.xpos[env.a_torso_id][2])
        com_x = float(np.sum(env._a_masses * env.data.xipos[env._a_bodies, 0]) / env._a_masses.sum())
        self._vx = 0.0 if self._com_prev is None else (com_x - self._com_prev) / 0.02
        self._com_prev = com_x
        if self.mode == "floor":
            use_ho = self.handover is not None and self.floor_name in self.handover_sides
            self._arrived = bool(use_ho and self.v_gate is not None and info.get("floor_success"))
            if not self._arrived:
                self._arrive_wait = 0
            if self._arrived:
                pass                                    # act() decides with the critic
            elif info.get("floor_success") and self._may_hand_over(env):
                self.mode = "handover" if (self.handover is not None and self.floor_name in self.handover_sides) else "stand"
                self.handovers += 1
                self._waited = 0
                self._vx_waited = 0
                self._clf_waited = 0
                self._ho_steps = self._ho_hold = 0
            elif z < FALL_HEIGHT:
                self._select_floor(env)        # re-pick by the current pitch sign while the body is down
        elif self.mode == "handover":
            from handover_env import FAIL_TORSO, HOLD_STEPS, ok_state
            self._ho_steps += 1
            self._ho_hold = self._ho_hold + 1 if ok_state(env) else 0
            if self._ho_hold >= HOLD_STEPS and self._ho_hold == HOLD_STEPS:
                self.ho_success += 1
                if not self.handover_forever:
                    self.mode = "stand"
            if self.mode != "handover":
                pass
            elif z < FAIL_TORSO:
                self.mode = "floor"; self.ho_fail += 1; self._floor_steps, self._flipped = 0, False; self._select_floor(env)
            elif self._ho_steps >= self.handover_max and not self.handover_forever:
                self.mode = "stand"          # time out: let the stand policy try anyway
        elif z < FALL_HEIGHT:
            self.falls += 1
            if self.fall_back:
                self.mode = "floor"
                self._floor_steps, self._flipped = 0, False
                self._select_floor(env)
