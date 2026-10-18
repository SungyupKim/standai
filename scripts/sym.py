"""Left-right symmetrisation of a stand policy (no retraining).

The 2D body's legs and arms are identical limbs, so swapping the right and left limbs (joints, velocities, actuators) maps a valid state to a valid state
with the same physics. The stand policy is NOT symmetric (it catches 73% of v12's hand-offs, which raise the LEFT leg, and 8% of their left-right swapped
copies, docs 3.22). SymPolicy asks the policy's own critic which of the two copies of the observation (as is / limbs swapped) it values more, acts on that
one and swaps the action back, so both versions of any hand-off are treated like the better-handled one.

obs permutation: found empirically (random states vs their limb-swapped copies; the dims that move are the shoulder/elbow/hip/knee/ankle positions and
velocities, the remaining dims -- parked opponent, target vx, COM terms -- are symmetric); action permutation from the actuator order
[waist, head, shoulder_r, elbow_r, shoulder_l, elbow_l, hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l] behind the unused action[0] (root x).
"""
import numpy as np
import torch
from stable_baselines3 import PPO

_OBS_PAIRS = [(5, 7), (6, 8), (9, 12), (10, 13), (11, 14), (20, 22), (21, 23), (24, 27), (25, 28), (26, 29)]
_ACT_PAIRS = [(3, 5), (4, 6), (7, 10), (8, 11), (9, 12)]          # action index = 1 + position in the actuator list above

OBS_PERM = np.arange(70)
for a, b in _OBS_PAIRS:
    OBS_PERM[a], OBS_PERM[b] = b, a
ACT_PERM = np.arange(13)
for a, b in _ACT_PAIRS:
    ACT_PERM[a], ACT_PERM[b] = b, a


def swap_obs(obs):
    return obs[..., OBS_PERM]


def swap_action(act):
    return act[..., ACT_PERM]


class SymPolicy:
    """Drop-in for PPO in code that only calls predict(obs, deterministic=True) and policy.predict_values."""

    def __init__(self, path_or_model, device="cpu"):
        self.model = PPO.load(path_or_model, device=device) if isinstance(path_or_model, str) else path_or_model
        self.policy = self.model.policy

    def _v(self, obs):
        with torch.no_grad():
            return float(self.policy.predict_values(torch.as_tensor(obs[None], dtype=torch.float32))[0, 0])

    def value(self, obs):
        return max(self._v(obs), self._v(swap_obs(obs)))

    def predict(self, obs, deterministic=True, **kw):
        use_swap = self._v(swap_obs(obs)) > self._v(obs)
        o = swap_obs(obs) if use_swap else obs
        act, state = self.model.predict(o, deterministic=deterministic, **kw)
        return (swap_action(act) if use_swap else act), state


def load_stand(path, device="cpu"):
    """The stand policy as the rest of the code should use it: symmetrised when STANDAI_SYM_STAND=1, plain PPO otherwise."""
    import os
    return SymPolicy(path, device=device) if os.environ.get("STANDAI_SYM_STAND") == "1" else PPO.load(path, device=device)
