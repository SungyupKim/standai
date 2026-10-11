"""Floor-only get-up environment: a separate, small problem from "stand and balance".

Every episode starts with the body lying/collapsed (torso < FLOOR_START_Z). The task is just to get the
head up to HANDOFF_HEAD_Z with a foot on the ground; reaching it ends the episode as a success and the
reached state is reported in info["handoff_state"] (qpos, qvel, head_z) so the stand policy can practise
from exactly the states this policy hands over. Reward is deliberately plain (no balance / stance terms):

    head_record  K * (new best normalised head height - previous best), foot on the floor only
    floor_dense  small per-step terms for head height / torso height with a foot down
    success      one-off bonus once the hand-off pose has been HELD for HOLD_STEPS consecutive steps

Hand-off pose = torso >= HOLD_TORSO_Z, head >= HANDOFF_HEAD_Z, a foot down, |pitch| < HANDOFF_MAX_PITCH_DEG.
(v1/v2 ended the episode the first instant head >= 0.9 m with a foot down; watching successes showed every one
of them dropping back to torso 0.3-0.5 m within 3 s, i.e. a pop-up, not a stand.)
"""
import numpy as np

from locomotion_env import LocomotionEnv

FLOOR_START_Z = 0.5            # resample the start until the torso is this low (lying / collapsed)
FLOOR_MAX_STEPS = 250          # 5 s, same as the env's own DOWN_RECOVERY_STEPS
HANDOFF_HEAD_Z = 0.9           # head height (m) at which the stand policy takes over
HANDOFF_MAX_PITCH_DEG = 40.0
HOLD_TORSO_Z = 0.6
HOLD_STEPS = 20                # 0.4 s of continuously holding the hand-off pose
RECORD_K = 10.0
DENSE_SCALE = 0.02
TORSO_DENSE_SCALE = 0.03
SUCCESS_BONUS = 10.0
_MAX_RESAMPLE = 30


class FloorEnv(LocomotionEnv):
    def __init__(self, **kwargs):
        super().__init__(stand_only=True, **kwargs)
        self._calm_frac = 0.0
        self._floor_head_best = 0.0
        self._floor_head_max = 0.0
        self._hold = 0

    def set_calm_frac(self, frac):          # the callbacks may broadcast this; a floor env is never calm
        self._calm_frac = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        for _ in range(_MAX_RESAMPLE):
            if self.data.xpos[self.a_torso_id][2] < FLOOR_START_Z:
                break
            obs, info = super().reset(options=options)
        head = float(self.data.xpos[self.a_head_id][2])
        self._floor_head_best = min(head, HANDOFF_HEAD_Z) / HANDOFF_HEAD_Z
        self._floor_head_max = head
        self._hold = 0
        return obs, info

    def step(self, action):
        obs, _parent_reward, terminated, truncated, info = super().step(action)
        head = float(self.data.xpos[self.a_head_id][2])
        foot_down, _ = self._getup_contacts()
        self._floor_head_max = max(self._floor_head_max, head)
        phi = min(head, HANDOFF_HEAD_Z) / HANDOFF_HEAD_Z

        record = 0.0
        if foot_down and phi > self._floor_head_best:
            record = RECORD_K * (phi - self._floor_head_best)
            self._floor_head_best = phi
        torso = float(self.data.xpos[self.a_torso_id][2])
        dense = (DENSE_SCALE * phi + TORSO_DENSE_SCALE * phi * min(torso / 0.7, 1.0)) * foot_down
        pitch = (self.data.qpos[self.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi
        in_pose = bool(foot_down and head >= HANDOFF_HEAD_Z and torso >= HOLD_TORSO_Z
                       and abs(pitch) < np.deg2rad(HANDOFF_MAX_PITCH_DEG))
        self._hold = self._hold + 1 if in_pose else 0
        success = self._hold >= HOLD_STEPS
        bonus = SUCCESS_BONUS if success else 0.0
        reward = record + dense + bonus

        info["reward_breakdown"] = {"head_record": record, "ground_height": dense}
        if success:
            terminated = True
            info["handoff_state"] = (self.data.qpos.copy(), self.data.qvel.copy(), head)
        if self.step_count >= FLOOR_MAX_STEPS:
            truncated = True
        if terminated or truncated:
            info["floor_success"] = success
            info["floor_head_max"] = self._floor_head_max
        return obs, reward, terminated, truncated, info
