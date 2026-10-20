"""Whole-system 20 s standing per side (benchmark.task_controller), with or without a hand-over policy, same seeds for both.
Usage: STANDAI_NO_RECOVERY_GATE=1 .venv/bin/python -I scripts/analysis/controller_side_eval.py --stand CKPT [--handover CKPT] [--n 100] [--seed0 0] [--workers 8]"""
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts"); sys.path.insert(0, ROOT + "scripts/analysis")
import benchmark as B

ap = argparse.ArgumentParser()
ap.add_argument("--stand", default=ROOT + "checkpoints/best_stand_v55b.zip")
ap.add_argument("--supine", default=B.DEFAULTS["supine"]); ap.add_argument("--prone", default=B.DEFAULTS["prone"])
ap.add_argument("--handover", nargs="*", default=[None], help="several checkpoints compare side by side ('none' = no hand-over policy)")
ap.add_argument("--v-gates", type=float, nargs="*", default=[None], help="hand-over critic gates to compare (omit = no gate)")
ap.add_argument("--flips", type=int, nargs="*", default=[0], help="floor_flip periods in steps to compare (0 = off)")
ap.add_argument("--forever", action="store_true", help="the hand-over policy keeps control instead of passing to the stand policy")
ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--levels", type=float, nargs="*", default=[0.7]); ap.add_argument("--sides", nargs="*", default=["supine", "prone"]); ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
with ProcessPoolExecutor(max_workers=a.workers) as ex:
    futs = {}
    for h in a.handover:
        h = None if h in (None, "none") else h
        for side in a.sides:
            for lv in a.levels:
                for vg in a.v_gates:
                    for fl in a.flips:
                        futs[(h, side, lv, vg, fl)] = ex.submit(B.task_controller, a.supine, a.prone, a.stand, a.n, side, lv, h, a.seed0, vg, fl, a.forever)
    for (h, side, lv, vg, fl), f in futs.items():
        r = f.result()
        print(f"hand-over {os.path.basename(h) if h else 'none':45s} {side:6s} level {lv} v_gate {vg} flip {fl}: standing at 20 s {r['standing_end']}/{r['n']}  (10 s {r['standing_10s']}, handed over {r['handed_over']})", flush=True)
