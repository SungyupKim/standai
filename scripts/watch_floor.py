"""Watches the floor get-up policy (floor_env.FloorEnv) in the MuJoCo viewer. Each episode starts lying
down; it ends when the head reaches the hand-off pose (success) or after 5 s. Deterministic actions.

--only-success: episodes are simulated headless (fast) and only successful ones are replayed in the viewer,
then the policy keeps running for --after-success seconds so you can see whether the pose is held.
"""
import argparse
import time

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from env import FRAME_SKIP
from floor_env import FloorEnv, HANDOFF_HEAD_Z


def run_episode(env, model, obs, on_step=None):
    """Runs one episode; returns (obs, info, qpos/qvel frames). on_step(env) is called after every step."""
    frames = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        frames.append((env.data.qpos.copy(), env.data.qvel.copy()))
        if on_step:
            on_step(env)
        if terminated or truncated:
            return obs, info, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds to hold the last frame between episodes")
    parser.add_argument("--after-success", type=float, default=3.0,
                        help="seconds to keep running the policy after a success (0 = end immediately)")
    parser.add_argument("--only-success", action="store_true",
                        help="simulate episodes headless and show only the successful ones")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device="cpu")
    env = FloorEnv()
    dt = env.model.opt.timestep * FRAME_SKIP
    n_ep = n_ok = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -10

        def show(_env):
            viewer.sync()
            time.sleep(dt)

        while viewer.is_running():
            obs, _ = env.reset()
            if args.only_success:
                obs, info, frames = run_episode(env, model, obs)        # headless
            else:
                obs, info, frames = run_episode(env, model, obs, on_step=show)
            n_ep += 1
            ok = bool(info["floor_success"])
            n_ok += ok
            print(f"episode {n_ep}: {'SUCCESS' if ok else 'fail'} "
                  f"(peak head {info['floor_head_max']:.2f} m, goal {HANDOFF_HEAD_Z} m, {len(frames)} steps) "
                  f"-- {n_ok}/{n_ep} so far", flush=True)
            if args.only_success:
                if not ok:
                    continue
                for qpos, qvel in frames:                                  # replay the recorded episode
                    if not viewer.is_running():
                        break
                    env.data.qpos[:], env.data.qvel[:] = qpos, qvel
                    mujoco.mj_forward(env.model, env.data)
                    show(env)
                env.data.qpos[:], env.data.qvel[:] = frames[-1]
                mujoco.mj_forward(env.model, env.data)
            if ok and args.after_success > 0:
                # the env ends the episode the instant the hand-off pose is reached; keep the same policy
                # running a few seconds to see whether that pose can be held
                for _ in range(int(args.after_success / dt)):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, *_ = env.step(action)
                    show(env)
                print(f"  after {args.after_success:.0f} s: torso {env.data.xpos[env.a_torso_id][2]:.2f} m, "
                      f"head {env.data.xpos[env.a_head_id][2]:.2f} m", flush=True)
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
