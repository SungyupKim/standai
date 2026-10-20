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
    parser.add_argument("--stand", type=str, default=None,
                        help="stand-policy checkpoint that takes over after a success (what the combined system does); "
                             "default: keep running the floor policy")
    parser.add_argument("--standready", action="store_true",
                        help="use the v6+ target (mixed lying/crouched starts, success = stand-ready region); needed for v6/v7/v8 checkpoints")
    parser.add_argument("--lying-only", action="store_true",
                        help="with --standready: no crouched starts (matches the v9+ training / notifier evaluation)")
    parser.add_argument("--prone-frac", type=float, default=0.0, help="share of lying starts forced to the start-pitch <= 0 side (1.0 = only those)")
    parser.add_argument("--supine-frac", type=float, default=0.0, help="share of lying starts forced to the start-pitch > 0 side (1.0 = supine only)")
    parser.add_argument("--no-down-timeout", action="store_true", help="disable env.py's 5 s 'stayed down' KO")
    parser.add_argument("--max-steps", type=int, default=None, help="episode length in steps (default 300 with --standready)")
    parser.add_argument("--skip-replay", action="store_true",
                        help="with --only-success: do not replay the get-up, show only what happens after the success (the stand policy's catch)")
    parser.add_argument("--only-success", action="store_true",
                        help="simulate episodes headless and show only the successful ones")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device="cpu")
    stand = PPO.load(args.stand, device="cpu") if args.stand else None
    env = FloorEnv(standready=args.standready, crouch_frac=0.0 if args.lying_only else 0.4,
                   max_steps=args.max_steps, prone_frac=args.prone_frac, supine_frac=args.supine_frac,
                   down_timeout=not args.no_down_timeout)
    dt = env.model.opt.timestep * FRAME_SKIP
    n_ep = n_ok = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -10

        track = {"zmax": 0.0, "air": 0, "n": 0}

        def show(_env):
            viewer.sync()
            time.sleep(dt)
            track["zmax"] = max(track["zmax"], float(_env.data.xpos[_env.a_torso_id][2]))
            track["n"] += 1
            track["air"] += 0 if _env._getup_contacts()[0] else 1     # steps with no foot on the floor

        while viewer.is_running():
            obs, _ = env.reset()
            track.update(zmax=0.0, air=0, n=0)
            if args.only_success:
                obs, info, frames = run_episode(env, model, obs)        # headless
            else:
                obs, info, frames = run_episode(env, model, obs, on_step=show)
            n_ep += 1
            ok = bool(info["floor_success"])
            n_ok += ok
            extra = ""
            if args.standready and track["n"]:
                extra = (f", start={info.get('start_kind', '?')}, peak torso {track['zmax']:.2f} m (need 1.08), "
                         f"no foot on floor {track['air'] / track['n']:.0%} of steps")
            print(f"episode {n_ep}: {'SUCCESS' if ok else 'fail'} "
                  f"(peak head {info['floor_head_max']:.2f} m, {len(frames)} steps{extra}) "
                  f"-- {n_ok}/{n_ep} so far", flush=True)
            if args.only_success:
                if not ok:
                    continue
                for qpos, qvel in ([] if args.skip_replay else frames):    # replay the recorded episode
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
                taker = stand if stand is not None else model
                min_torso = 9.0
                for _ in range(int(args.after_success / dt)):
                    action, _ = taker.predict(obs, deterministic=True)
                    obs, *_ = env.step(action)
                    min_torso = min(min_torso, float(env.data.xpos[env.a_torso_id][2]))
                    show(env)
                tz = float(env.data.xpos[env.a_torso_id][2])
                print(f"  {'stand policy' if stand is not None else 'floor policy'} for {args.after_success:.0f} s: "
                      f"torso {tz:.2f} m (min {min_torso:.2f}), head {env.data.xpos[env.a_head_id][2]:.2f} m -> "
                      f"{'HELD' if tz >= 0.7 else 'FELL'}", flush=True)
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
