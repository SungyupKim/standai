"""Hand-over (catch) policy environment: from a hand-off state (the floor policy's arrival) to a state the STAND policy holds.

The stand policy only catches states whose centre of mass is behind the front foot (prone hand-offs: 84% below -5 cm ahead, 0% above +10 cm;
docs 3.22/3.23). This policy is trained on those hand-off states (checkpoints/pool_big_all.npz through the replay pool) to move the body into
the stand policy's region by itself -- stepping, leaning, whatever it learns -- and then hands over.

Observation / action = the stand policy's (LocomotionEnv, stand_only), so scripts/sym.py and every stand-side tool apply unchanged.
Success (ok for `hold_steps` in a row, the episode ends and the controller switches to the stand policy): a foot down, torso >= OK_TORSO,
|pitch| < OK_PITCH, COM at least OK_BACK behind the front foot, slow (|COM vx|, pitch rate). Failure: torso below FAIL_TORSO (collapsed).
Reward: potential-based shaping on (COM excess over the support + torso-height deficit), paid only while the body is up and not tipped over,
a small time cost (no per-step upright pay: ending the episode on success must beat surviving), a success bonus and a fall penalty.
"""
import os

import numpy as np

from locomotion_env import LocomotionEnv

OK_TORSO = float(os.environ.get("STANDAI_HO_TORSO", "1.0"))
OK_PITCH_DEG = float(os.environ.get("STANDAI_HO_PITCH", "25"))
OK_BACK = float(os.environ.get("STANDAI_HO_BACK", "0.03"))        # COM x must be <= front foot x - OK_BACK
REAR_MARGIN = 0.08
OK_VX = float(os.environ.get("STANDAI_HO_VX", "0.4"))
OK_PR_DEG = float(os.environ.get("STANDAI_HO_PR", "80"))
HOLD_STEPS = 10
GROUND_TOL = 0.06                                                  # a foot within this height of the lower one counts as grounded
FAIL_TORSO = 0.45
SHAPE_K = float(os.environ.get("STANDAI_HO_K", "100"))            # weight of the COM-excess potential
HEIGHT_W = float(os.environ.get("STANDAI_HO_HEIGHT_W", "1.0"))      # torso-height deficit (below OK_TORSO) counts like COM excess inside the potential
TIME_COST = 0.02
SUCCESS_BONUS = 60.0
FALL_PENALTY = 20.0
MAX_STEPS = 300



def com_state(env):
    """(COM x, front GROUNDED foot x, rear GROUNDED foot x)."""
    m = env._a_masses
    com_x = float(np.sum(m * env.data.xipos[env._a_bodies, 0]) / m.sum())
    fx = [float(env.data.site_xpos[s][0]) for s in env.a_feet]
    fz = [float(env.data.site_xpos[s][2]) for s in env.a_feet]
    grounded = [x for x, z in zip(fx, fz) if z <= min(fz) + GROUND_TOL]     # a raised foot is no support (supine hand-offs raise one leg)
    return com_x, max(grounded), min(grounded)


def support_excess(env):
    """How far (m) the COM is outside the support interval [rear grounded foot - REAR_MARGIN, front grounded foot - OK_BACK]; 0 = inside.
    Behind a single support foot is as bad as ahead of it (supine hand-offs stand on one leg), so both sides count."""
    com_x, front, rear = com_state(env)
    return max(0.0, com_x - (front - OK_BACK)) + max(0.0, (rear - REAR_MARGIN) - com_x)


def ok_state(env):
    """The hand-over success condition (one step); works on any stand-style env (LocomotionEnv / FloorEnv / HandoverEnv)."""
    foot_down, _ = env._getup_contacts()
    torso = float(env.data.xpos[env.a_torso_id][2])
    pitch = float((env.data.qpos[env.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi)
    vx = float(env.data.qvel[env._a_root_x_dof])
    pr = float(np.degrees(env.data.qvel[env._ry_dof]))
    return bool(foot_down and torso >= OK_TORSO and abs(pitch) < np.deg2rad(OK_PITCH_DEG)
                and support_excess(env) <= 0.0 and abs(vx) < OK_VX and abs(pr) < OK_PR_DEG)


class HandoverEnv(LocomotionEnv):
    def __init__(self, max_steps=MAX_STEPS, **kwargs):
        kwargs.pop("stand_only", None)
        super().__init__(stand_only=True, **kwargs)
        self._ho_max_steps = max_steps
        self._ho_hold = 0
        self._ho_prev_phi = None
        self._ry_dof = self.model.joint("a_root_ry").dofadr[0]

    def set_calm_frac(self, frac):          # callbacks may broadcast this; every episode starts from a hand-off state
        pass

    def com_state(self):
        return com_state(self)

    def support_excess(self):
        return support_excess(self)

    def ok_state(self):
        return ok_state(self)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._ho_hold = 0
        self._ho_prev_phi = None
        return obs, info

    def step(self, action):
        obs, _r, terminated, truncated, info = super().step(action)
        torso = float(self.data.xpos[self.a_torso_id][2])
        pitch = float((self.data.qpos[self.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi)
        up = torso >= 0.6 and abs(pitch) < np.deg2rad(40.0)
        shape = 0.0
        if up:
            phi = -min(self.support_excess(), 0.4) - HEIGHT_W * min(max(0.0, OK_TORSO - torso), 0.5)
            if self._ho_prev_phi is not None:
                shape = SHAPE_K * (0.99 * phi - self._ho_prev_phi)
            self._ho_prev_phi = phi
        else:
            self._ho_prev_phi = None
        self._ho_hold = self._ho_hold + 1 if self.ok_state() else 0
        success = self._ho_hold >= HOLD_STEPS
        failed = torso < FAIL_TORSO
        reward = shape - TIME_COST + (SUCCESS_BONUS if success else 0.0) - (FALL_PENALTY if failed else 0.0)
        terminated = bool(success or failed)
        truncated = bool(self.step_count >= self._ho_max_steps and not terminated)
        info["reward_breakdown"] = {"step": shape}
        if terminated or truncated:
            info["ho_success"] = bool(success)
            info["ho_fail"] = bool(failed)
        if success:
            info["handoff_state"] = (self.data.qpos.copy(), self.data.qvel.copy(), float(self.data.xpos[self.a_head_id][2]))
        return obs, reward, terminated, truncated, info
