"""Plays only the FAILURE cases of the whole get-up system in the MuJoCo viewer.
First a headless scan (--scan seeds) finds the starts that do NOT end standing after --seconds (standing = torso >= 0.95 m and |pitch| < 30 deg for the last 25 steps,
the benchmark's definition) with the full controller (floor policy -> optional hand-over policy -> stand policy, GetUpController). Then the first --show failures are
replayed in the viewer; the console says what kind of failure each one is:
  never reached   the floor policy never reached the hand-over region
  reached, down   handed over but the body ended on the floor
  stood, fell     was upright at some time and ended down again
Usage: DISPLAY=:1 ../.venv/bin/python watch_failures.py [--side supine|prone] [--seed0 17000] [--scan 40] [--show 10] [--handover CKPT] [--seconds 20]
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from controller import GetUpController
from env import FRAME_SKIP
from floor_env import FloorEnv

A = "../checkpoints/"
SUPINE = A + "autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip"
PRONE = A + "best_prone8.zip"
STAND = A + "best_stand_v55b.zip"
HO_PRONE = A + "best_handover_prone.zip"


def upright(env, ctl):
    return float(env.data.xpos[env.a_torso_id][2]) >= 0.95 and abs(np.degrees(ctl.pitch(env))) < 30


def episode(env, ctl, seed, steps, viewer=None, dt=0.0, verbose=False):
    obs, _ = env.reset(seed=seed)
    ctl.reset(env)
    last, hist, stood = ctl.mode, [], False
    for t in range(steps):
        if viewer is not None and not viewer.is_running():
            return None
        obs, _, term, trunc, info = env.step(ctl.act(obs, env))
        ctl.observe(env, info)
        hist.append(upright(env, ctl))
        stood = stood or (len(hist) >= 25 and all(hist[-25:]))
        if verbose and ctl.mode != last:
            print(f"    t={t * dt:5.2f}s  {last} -> {ctl.mode}   (torso {env.data.xpos[env.a_torso_id][2]:.2f} m)", flush=True)
            last = ctl.mode
        if viewer is not None:
            viewer.sync()
            time.sleep(dt)
    ok = len(hist) >= 25 and all(hist[-25:])
    kind = "standing" if ok else ("stood, fell" if stood else ("never reached" if ctl.handovers == 0 else "reached, down"))
    return ok, kind, float(env.data.xpos[env.a_torso_id][2]), float(np.degrees(ctl.pitch(env)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["supine", "prone"], default="supine")
    p.add_argument("--level", type=float, default=0.7)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--seed0", type=int, default=17000)
    p.add_argument("--scan", type=int, default=40, help="starts to simulate headless when looking for failures")
    p.add_argument("--show", type=int, default=10, help="how many failures to replay in the viewer")
    p.add_argument("--supine", default=SUPINE)
    p.add_argument("--prone", default=PRONE)
    p.add_argument("--stand", default=STAND)
    p.add_argument("--handover", default=None, help="hand-over policy checkpoint (default: best_handover_prone for the prone side, none for the supine side)")
    p.add_argument("--pause", type=float, default=2.0)
    p.add_argument("--speed", type=float, default=1.0)
    a = p.parse_args()

    ho = a.handover if a.handover else (HO_PRONE if a.side == "prone" else None)
    env = FloorEnv(standready=True, level=a.level, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False,
                   prone_frac=1.0 if a.side == "prone" else 0.0, supine_frac=1.0 if a.side == "supine" else 0.0)
    ctl = GetUpController(a.supine, a.prone, a.stand, fall_back=True, handover_ckpt=ho, handover_sides=(a.side,))
    steps = int(a.seconds / (env.model.opt.timestep * FRAME_SKIP))
    dt = env.model.opt.timestep * FRAME_SKIP / a.speed

    fails, counts = [], {}
    print(f"scanning {a.scan} {a.side} starts (seeds {a.seed0}..{a.seed0 + a.scan - 1}) headless ...", flush=True)
    for s in range(a.seed0, a.seed0 + a.scan):
        ok, kind, z, pit = episode(env, ctl, s, steps)
        counts[kind] = counts.get(kind, 0) + 1
        if not ok:
            fails.append((s, kind))
    print(f"result: {counts}  -> {len(fails)} failures of {a.scan}", flush=True)
    if not fails:
        return
    print("failure seeds:", [f"{s}:{k}" for s, k in fails], flush=True)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -10
        n = 0
        while viewer.is_running():
            s, kind = fails[n % min(len(fails), a.show)]
            print(f"failure {n % min(len(fails), a.show) + 1}/{min(len(fails), a.show)}: seed {s} ({kind})", flush=True)
            r = episode(env, ctl, s, steps, viewer=viewer, dt=dt, verbose=True)
            if r is None:
                break
            print(f"    ended: {r[1]} (torso {r[2]:.2f} m, pitch {r[3]:+.0f} deg)", flush=True)
            n += 1
            time.sleep(a.pause)


if __name__ == "__main__":
    main()
