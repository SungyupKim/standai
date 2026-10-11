"""Watches the solo mobility/recovery policy (locomotion_env.LocomotionEnv) in the
interactive MuJoCo viewer -- 'b' is parked off to the side and irrelevant, no health
bars/hit markers since there's no combat. Resets are triggered on a timer as well as on
episode end, so a checkpoint that's gotten good at NOT falling still gets shown a fresh
random perturbation regularly instead of just standing still forever.
"""
import argparse
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from env import FALL_HEIGHT, FRAME_SKIP
from locomotion_env import LocomotionEnv

RESET_EVERY_SECONDS = 20.0     # matches env.py's MAX_STEPS=1000 (*0.02s/step) -- 6.0 was
                                # cutting episodes off at <40% of the ~700-750 step ep_len_mean
                                # measured at the end of training, hiding whatever happens after
                                # perturbation-recovery (docs 10.5x)


POSE_EPS = 0.03   # per-coordinate range (rad for joints, m for slides) below which the pose counts as unchanged


class StuckDetector:
    """Flags an episode whose pose hasn't changed over a sliding window: every 'a_' joint/root
    coordinate stayed within POSE_EPS. On the ground the window is `ground_s` seconds; standing
    still is a success, so it gets a longer `stand_s` before moving on (0 = never skip on this)."""

    def __init__(self, env, ground_s, stand_s):
        m = env.model
        self.adr = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
                    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '').startswith('a_')]
        step_s = m.opt.timestep * FRAME_SKIP
        self.ground_n = int(round(ground_s / step_s)) if ground_s > 0 else 0
        self.stand_n = int(round(stand_s / step_s)) if stand_s > 0 else 0
        self.hist = deque(maxlen=max(self.ground_n, self.stand_n, 1))

    def reset(self):
        self.hist.clear()

    def update(self, env):
        """Record this step's pose; returns a reason string if the episode should be skipped."""
        self.hist.append(env.data.qpos[self.adr].copy())
        standing = env.data.xpos[env.a_torso_id][2] >= FALL_HEIGHT
        n = self.stand_n if standing else self.ground_n
        if n <= 0 or len(self.hist) < n:
            return None
        window = np.array(list(self.hist)[-n:])
        if np.ptp(window, axis=0).max() < POSE_EPS:
            return f"no pose change for {n * env.model.opt.timestep * FRAME_SKIP:.1f}s ({'standing' if standing else 'on the ground'})"
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    # bug found 2026-10-09: this was never wired up at all, so every stand-only checkpoint
    # watched here all night was actually running with stand_only=False -- BALANCE_KP/KD (the
    # torso-pitch assist that's supposed to be off) and the horizontal thrust actuator (whose
    # action dim training always zeroed, so its real output here is untrained noise) were both
    # silently active, making everything on screen easier than what was actually trained/gated.
    # Defaults to True since every checkpoint in active use right now is stand-only; pass
    # --no-stand-only for an older walking-mode checkpoint (e.g. docs section 10's v12fresh2).
    parser.add_argument("--no-stand-only", dest="stand_only", action="store_false", default=True,
                         help="watch a non-stand-only (walking-mode) checkpoint instead")
    parser.add_argument("--knee-floor-deg", type=float, default=-80.0,
                         help="match the knee-angle scaffold's permanent floor during calm "
                              "episodes (set to None-equivalent via a very negative number, "
                              "e.g. -140, to disable)")
    parser.add_argument("--stuck-seconds", type=float, default=2.0,
                         help="on the ground (torso < FALL_HEIGHT): move to the next episode after this "
                              "many seconds without a pose change (0 disables)")
    parser.add_argument("--stand-still-seconds", type=float, default=5.0,
                         help="standing (torso >= FALL_HEIGHT): move on after this many seconds without a "
                              "pose change (0 disables; the 20s reset timer still applies)")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device="cpu")
    env = LocomotionEnv(stand_only=args.stand_only)
    if args.stand_only:
        env.set_min_knee_deg(args.knee_floor_deg)
    obs, info = env.reset()
    last_reset = time.time()
    stuck = StuckDetector(env, args.stuck_seconds, args.stand_still_seconds)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 1.0]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -10

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(env.model.opt.timestep * FRAME_SKIP)

            reason = stuck.update(env)
            if terminated or truncated or reason or (time.time() - last_reset > RESET_EVERY_SECONDS):
                a_out = info.get("a_out")
                why = reason or f"terminated={terminated} truncated={truncated} a_out={a_out}"
                print(f"reset ({why})", flush=True)
                obs, info = env.reset()
                last_reset = time.time()
                stuck.reset()


if __name__ == "__main__":
    main()
