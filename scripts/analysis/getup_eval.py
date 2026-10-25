"""Paired measurement of the end-to-end student(s) against the rule-based controller on the same fresh seeds: standing at 10 s / 20 s per start side (the controller benchmark's criterion).
Usage: .venv/bin/python -I scripts/analysis/getup_eval.py [--arms teacher ../checkpoints/getup_student_distilled.zip ...] [--sets 91000 92000] [--n 200] [--workers 8]"""
import argparse, os, sys
from multiprocessing import get_context
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
from distill_controller import eval_pool, fmt

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["teacher", ROOT + "checkpoints/getup_student_distilled.zip"])
ap.add_argument("--sets", type=int, nargs="+", default=[91000, 92000])
ap.add_argument("--n", type=int, default=200); ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
tot = {}
with get_context("fork").Pool(a.workers) as pool:
    for arm in a.arms:
        for s in a.sets:
            r = eval_pool(pool, None if arm == "teacher" else arm, a.n, s, a.workers)
            print(f"{os.path.basename(arm):32s} seeds {s}+: {fmt(r)}", flush=True)
            t = tot.setdefault(arm, [0, 0, 0, 0, 0]); t[0] += r["supine"][1]; t[1] += r["supine"][2]; t[2] += r["prone"][1]; t[3] += r["prone"][2]
for arm, t in tot.items():
    print(f"TOTAL {os.path.basename(arm):32s} supine {t[0]}/{t[1]} ({t[0] / t[1]:.1%}), prone {t[2]}/{t[3]} ({t[2] / t[3]:.1%}), 50:50 {(t[0] / t[1] + t[2] / t[3]) / 2:.1%}")
