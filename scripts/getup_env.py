"""GetUpEnv: the whole get-up task as ONE environment for a single end-to-end policy (docs 3.24 addendum 13).

A lying start (supine or prone, 50:50 per reset, or a fixed side) in the same FloorEnv configuration the controller is benchmarked in (stand-ready region level 0.7, no down timeout), a fixed horizon
(default 1000 steps = 20 s) and NO termination at the hand-over: the episode continues while the policy stands, falls and gets up again. Reward per step = 1 while standing (torso >= 0.95 m and
|pitch| < 30 deg, the benchmark's 'standing' criterion) + 0.1 * clip((torso - 0.5) / 0.45, 0, 1) as a weak height shaping so lying starts are not pure zeros. FloorEnv's `info["floor_success"]` is kept
so the rule-based GetUpController (the teacher) can run on the same environment."""
import numpy as np

from floor_env import FloorEnv


class GetUpEnv(FloorEnv):
    def __init__(self, horizon=1000, side=None, level=0.7, **kw):
        super().__init__(standready=True, level=level, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False, **kw)
        self._horizon = int(horizon)
        self._fixed_side = side
        self._side_rng = np.random.default_rng()
        self._t = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._side_rng = np.random.default_rng(seed + 7919)
        side = self._fixed_side or ("prone" if self._side_rng.random() < 0.5 else "supine")
        self._prone_frac, self._supine_frac = (1.0, 0.0) if side == "prone" else (0.0, 1.0)
        self._t = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, _r, _term, _trunc, info = super().step(action)
        self._t += 1
        z = float(self.data.xpos[self.a_torso_id][2])
        pitch = (self.data.qpos[self.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi
        standing = z >= 0.95 and abs(float(np.degrees(pitch))) < 30.0
        reward = float(standing) + 0.1 * float(np.clip((z - 0.5) / 0.45, 0.0, 1.0))
        info["standing"] = standing
        return obs, reward, False, self._t >= self._horizon, info
