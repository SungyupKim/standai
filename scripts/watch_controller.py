"""Watches the whole get-up system in the MuJoCo viewer: lying start -> floor policy (supine or prone, picked by the start pitch sign)
-> hand-over to the stand policy -> fall back to the floor policy if the stand policy falls (GetUpController, scripts/controller.py).
Each episode runs --seconds (default 20 s, the benchmark's horizon); the console prints the mode changes and whether the body is standing at the end
(torso >= 0.95 m and |pitch| < 30 deg, same definition as benchmark.py).

DISPLAY=:1 ../.venv/bin/python watch_controller.py [--side supine|prone|mixed] [--level 0.7] [--seed 0]
Defaults: v12 / prone8 / best_stand_v55b + best_handover_prone (prone side only), hand-over level 0.7, supine and prone starts alternate ('mixed' = the env's natural 80:20 draw).
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from controller import GetUpController
from env import FRAME_SKIP
from floor_env import FloorEnv
from viewer_utils import hide_opponent

A = "../checkpoints/"
SUPINE = A + "autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip"
PRONE = A + "best_prone8.zip"
STAND = A + "best_stand_v55b.zip"
HANDOVER = A + "best_handover_prone.zip"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["supine", "prone", "alternate", "mixed"], default="alternate")
    p.add_argument("--level", type=float, default=0.7, help="hand-over region: 1.0 strict ... 0.0 loose")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--supine", default=SUPINE)
    p.add_argument("--prone", default=PRONE)
    p.add_argument("--stand", default=STAND)
    p.add_argument("--handover", default=HANDOVER, help="hand-over policy used after the PRONE floor policy ('none' = off)")
    p.add_argument("--seed", type=int, default=0, help="first reset seed; increments per episode")
    p.add_argument("--pause", type=float, default=1.5)
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1.0 = real time)")
    a = p.parse_args()

    env = FloorEnv(standready=True, level=a.level, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False, prone_frac=0.0, supine_frac=0.0)
    ho = None if a.handover == "none" else a.handover
    ctl = GetUpController(a.supine, a.prone, a.stand, fall_back=True, handover_ckpt=ho, handover_sides=("prone",))
    dt = env.model.opt.timestep * FRAME_SKIP / a.speed
    steps = int(a.seconds / (env.model.opt.timestep * FRAME_SKIP))
    n_ep = n_ok = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        hide_opponent(env.model, viewer)
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -10
        seed = a.seed
        while viewer.is_running():
            side = a.side
            if side == "alternate":
                side = "supine" if n_ep % 2 == 0 else "prone"
            env._prone_frac = 1.0 if side == "prone" else 0.0
            env._supine_frac = 1.0 if side == "supine" else 0.0
            obs, _ = env.reset(seed=seed)
            seed += 1
            ctl.reset(env)
            last_mode, hist = ctl.mode, []
            t0 = time.time()
            for t in range(steps):
                if not viewer.is_running():
                    break
                obs, _, term, trunc, info = env.step(ctl.act(obs, env))
                ctl.observe(env, info)
                hist.append((float(env.data.xpos[env.a_torso_id][2]), ctl.pitch(env)))
                if ctl.mode != last_mode:
                    print(f"  t={t * dt * a.speed:5.2f}s  {last_mode} -> {ctl.mode}   (torso {hist[-1][0]:.2f} m)", flush=True)
                    last_mode = ctl.mode
                viewer.sync()
                time.sleep(max(0.0, dt - 0.0))
            ok = len(hist) >= 25 and all(z >= 0.95 and abs(np.degrees(pp)) < 30 for z, pp in hist[-25:])
            n_ep += 1
            n_ok += ok
            print(f"episode {n_ep} ({ctl.floor_name} start, seed {seed - 1}): {'STANDING' if ok else 'not standing'} at the end "
                  f"(hand-overs {ctl.handovers}, stand-policy falls {ctl.falls}) -- {n_ok}/{n_ep} so far", flush=True)
            time.sleep(a.pause)


if __name__ == "__main__":
    main()
