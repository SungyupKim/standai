"""Renders a demo video of the whole get-up system (floor policy -> hand-over policy (prone side) -> stand policy) offscreen, no display needed.

Picks the first --supine-ok / --prone-ok starts (seeds from --seed0) that END STANDING after 20 s without ever falling on the way (--any-success lifts that) (benchmark definition: torso >= 0.95 m and |pitch| < 30 deg for the last
25 steps), re-runs them (the policies are deterministic) and records the first --seconds of each. These are SELECTED SUCCESSES: the caption says so and quotes the overall rate.
Writes assets/demo.mp4 (640x360, 25 fps) and assets/demo.gif (480x270, 12 fps). Needs ffmpeg and PIL.
Usage: python scripts/render_demo.py [--supine-ok 2] [--prone-ok 2] [--seed0 31000] [--seconds 10] [--out assets/demo]
"""
import argparse
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STANDAI_NO_RECOVERY_GATE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from controller import GetUpController
from env import FRAME_SKIP
from floor_env import FloorEnv

A = os.path.join(HERE, "..", "checkpoints") + "/"
SUPINE = A + "best_floor_supine_fdist.zip"
PRONE = A + "best_prone8.zip"
STAND = A + "best_stand_big1.zip"
HANDOVER = A + "best_handover_prone.zip"
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
SCENE_OPT = mujoco.MjvOption()
SCENE_OPT.geomgroup[5] = 0      # hidden group: the parked opponent
CLEAN_ONLY = True      # pick only runs that never fell on the way (see run())
CLEAN_BY = 8.0         # seconds: must be standing by then
MODE_LABEL = {"floor": ("Floor policy", (255, 200, 80)), "handover": ("Hand-over policy", (120, 200, 255)), "stand": ("Stand policy", (120, 230, 140))}


def font(size):
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def hide_opponent(model):
    """The unused fighting opponent ('b', blue) is parked at x = 6 m; the tracking camera shows it once the body walks there. Move its geoms to a hidden visualization group
    (scene option below) and make them fully transparent so they cast no shadow either."""
    for g in range(model.ngeom):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").startswith("b_"):
            model.geom_group[g] = 5
            model.geom_rgba[g, 3] = 0.0


def make(side):
    env = FloorEnv(standready=True, level=0.7, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False,
                   prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    ctl = GetUpController(SUPINE, PRONE, STAND, fall_back=True, handover_ckpt=HANDOVER, handover_sides=("prone",))
    return env, ctl


def run(env, ctl, seed, steps, on_frame=None):
    obs, _ = env.reset(seed=seed)
    ctl.reset(env)
    hist, first_up, up_once, dipped, head_peak = [], None, False, False, 0.0
    for t in range(steps):
        if on_frame is not None:
            on_frame(t)
        obs, _, term, trunc, info = env.step(ctl.act(obs, env))
        ctl.observe(env, info)
        th = float(env.data.xpos[env.a_torso_id][2])
        hd = float(env.data.xpos[env.a_head_id][2])
        head_peak = max(head_peak, hd)
        dipped = dipped or (head_peak >= 0.7 and hd < head_peak - 0.35)    # got the head up, then dropped back down (sat/fell back): not a clean rise
        up_once = up_once or th >= 0.9
        dipped = dipped or (up_once and th < 0.75)    # stood up once, then sank back / fell: not a clean run
        hist.append(float(env.data.xpos[env.a_torso_id][2]) >= 0.95 and abs(np.degrees(ctl.pitch(env))) < 30)
        if first_up is None and len(hist) >= 25 and all(hist[-25:]):
            first_up = t * 0.02
    ok = len(hist) >= 25 and all(hist[-25:])
    # clean = it never fell on the way: no stand-policy fall, no hand-over collapse, and it was up early enough to be seen in the clip
    clean = ok and not dipped and ctl.falls == 0 and ctl.ho_fail == 0 and first_up is not None and first_up <= CLEAN_BY
    return clean if CLEAN_ONLY else ok


def overlay(img, side, mode, t, caption):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    W, H = im.size
    d.rectangle([0, 0, W, 34], fill=(0, 0, 0))
    d.text((10, 6), f"{side.capitalize()} start ({'lying on the back' if side == 'supine' else 'lying face down'})", font=font(17), fill=(255, 255, 255))
    d.text((W - 90, 8), f"{t:4.1f} s", font=font(16), fill=(220, 220, 220))
    label, col = MODE_LABEL[mode]
    d.rectangle([10, 42, 10 + 12 + 10 * len(label), 70], fill=(0, 0, 0))
    d.text((16, 46), label, font=font(17), fill=col)
    d.rectangle([0, H - 24, W, H], fill=(0, 0, 0))
    d.text((8, H - 21), caption, font=font(12), fill=(230, 230, 230))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supine-ok", type=int, default=2)
    ap.add_argument("--prone-ok", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=31000)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "assets", "demo"))
    ap.add_argument("--any-success", action="store_true", help="also allow runs that fell and got up again on the way")
    ap.add_argument("--caption", default="Selected successful runs (simulation). Overall ~65% stand after 20 s (200-400 seeds/side).")
    a = ap.parse_args()
    global CLEAN_ONLY
    CLEAN_ONLY = not a.any_success

    steps20 = int(20.0 / (0.005 * FRAME_SKIP))
    picks = []
    for side, need in (("supine", a.supine_ok), ("prone", a.prone_ok)):
        env, ctl = make(side)
        seed, got, tried = a.seed0, 0, 0
        while got < need and tried < 120:
            ok = run(env, ctl, seed, steps20)
            tried += 1
            if ok:
                picks.append((side, seed)); got += 1
            seed += 1
        print(f"{side}: {got} successes in {tried} starts tried (seeds {a.seed0}..{seed - 1})", flush=True)
    # interleave supine / prone
    order = []
    s = [p for p in picks if p[0] == "supine"]; p_ = [p for p in picks if p[0] == "prone"]
    for i in range(max(len(s), len(p_))):
        if i < len(s): order.append(s[i])
        if i < len(p_): order.append(p_[i])

    tmp = tempfile.mkdtemp(prefix="demo_frames_")
    n_frame = 0
    renderer = None
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.4, -90, -8
    for side, seed in order:
        env, ctl = make(side)
        if renderer is None:
            renderer = mujoco.Renderer(env.model, height=360, width=640)
        hide_opponent(env.model)
        state = {"x": None}

        def on_frame(t):
            nonlocal n_frame
            if t % 2 or t * 0.02 > a.seconds:
                return
            x = float(env.data.xpos[env.a_torso_id][0])
            state["x"] = x if state["x"] is None else 0.9 * state["x"] + 0.1 * x      # smoothed horizontal tracking
            cam.lookat[:] = [state["x"], 0.0, 0.7]
            renderer.update_scene(env.data, camera=cam, scene_option=SCENE_OPT)
            frame = overlay(renderer.render(), side, ctl.mode, t * 0.02, a.caption)
            Image.fromarray(frame).save(f"{tmp}/f{n_frame:05d}.png")
            n_frame += 1

        ok = run(env, ctl, seed, steps20, on_frame=on_frame)
        print(f"clip {side} seed {seed}: ends standing = {ok}, frames so far {n_frame}", flush=True)
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", out + ".mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i", f"{tmp}/f%05d.png", "-vf", "fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4", out + ".gif"], check=True)
    for ext in (".mp4", ".gif"):
        print(f"{out + ext}: {os.path.getsize(out + ext) / 1e6:.1f} MB", flush=True)
    subprocess.run(["rm", "-rf", tmp])


if __name__ == "__main__":
    main()
