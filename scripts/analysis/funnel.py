"""Whole-system funnel per side, same seeds, 20 s episodes (GetUpController, hand-over region --level): where do the starts that do not end up standing get lost?
  reached     the floor policy reached the hand-over region at least once
  ho ok       (with a hand-over policy) it reached its success state at least once / collapsed at least once
  stood       was upright (torso >= 0.95 m, |pitch| < 30 deg) for 25 steps in a row at some time / at 10 s / at 20 s
  falls       times the stand policy lost the torso below FALL_HEIGHT and control went back to the floor policy (mean per start)
  end state   standing | never reached the region | reached but ended down | stood once but ended down
Usage: STANDAI_NO_RECOVERY_GATE=1 .venv/bin/python -I scripts/analysis/funnel.py [--stand CKPT] [--ho-prone CKPT] [--ho-supine CKPT] [--n 100] [--seed0 7000] [--level 0.7]"""
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts"); sys.path.insert(0, ROOT + "scripts/analysis")
import numpy as np


def run_side(side, stand, supine, prone, ho, n, seed0, level):
    import benchmark as B
    from controller import GetUpController
    from floor_env import FloorEnv
    env = FloorEnv(standready=True, level=level, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False,
                   prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    ctl = GetUpController(supine, prone, stand, fall_back=True, handover_ckpt=ho, handover_sides=(side,))
    rows = []
    for seed in range(seed0, seed0 + n):
        obs, _ = env.reset(seed=seed); ctl.reset(env)
        ok_hist, first_ho, ever = [], None, False
        for t in range(1000):
            obs, _, term, trunc, info = env.step(ctl.act(obs, env)); ctl.observe(env, info)
            if first_ho is None and ctl.handovers > 0: first_ho = t
            ok_hist.append(float(env.data.xpos[env.a_torso_id][2]) >= 0.95 and abs(np.degrees(ctl.pitch(env))) < 30)
        h = np.array(ok_hist)
        run = np.convolve(h, np.ones(25), mode="valid") >= 25          # 25-step windows fully upright
        stood_ever, at10, at20 = bool(run.any()), bool(run[475 - 0]) if len(run) > 475 else False, bool(run[-1])
        rows.append((first_ho is not None, ctl.ho_success > 0, ctl.ho_fail > 0, stood_ever, at10, at20, ctl.falls, first_ho if first_ho is not None else -1))
    return side, np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stand", default=ROOT + "checkpoints/best_stand_v55b.zip")
    ap.add_argument("--ho-prone", default=None); ap.add_argument("--ho-supine", default=None)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed0", type=int, default=7000); ap.add_argument("--level", type=float, default=0.7)
    a = ap.parse_args()
    import benchmark as B
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run_side, s, a.stand, B.DEFAULTS["supine"], B.DEFAULTS["prone"], h, a.n, a.seed0, a.level)
                for s, h in (("supine", a.ho_supine), ("prone", a.ho_prone))]
        res = dict(f.result() for f in futs)
    for side, R in res.items():
        n = len(R); reached, ho_ok, ho_bad, ever, a10, a20, falls, t_ho = R.T
        ended_up = a20 == 1
        never = reached == 0
        print(f"== {side} ({n} starts, hand-over policy: {'yes' if (a.ho_prone if side == 'prone' else a.ho_supine) else 'no'}) ==")
        print(f"  reached the hand-over region   {reached.mean():4.0%}   (median first hand-over at step {np.median(t_ho[reached == 1]):.0f} = {np.median(t_ho[reached == 1]) / 50:.1f} s)")
        if ho_ok.any() or ho_bad.any():
            print(f"  hand-over policy success / collapse   {ho_ok.mean():4.0%} / {ho_bad.mean():4.0%}  (of starts, at least once)")
        print(f"  stood at some time {ever.mean():4.0%} | at 10 s {a10.mean():4.0%} | at 20 s {a20.mean():4.0%}   stand-policy falls per start {falls.mean():.2f}")
        print(f"  end state: standing {ended_up.mean():4.0%} | never reached the region {never.mean():4.0%} | reached, ended down {np.mean((reached == 1) & (ever == 0)):4.0%} "
              f"| stood once, ended down {np.mean((ever == 1) & ~ended_up):4.0%}")


if __name__ == "__main__":
    main()
