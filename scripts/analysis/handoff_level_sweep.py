"""Hand over to the stand policy at the first entry into a LOOSER stand-ready region (FloorEnv level L: 1.0 = strict). The floor policy is the same
(trained for the strict region); only the trigger changes. Reports reached and held-by-stand (3 s, never below 0.7 m) per level.
Usage: handoff_level_sweep.py <floor_ckpt> <n> <prone_frac> <levels comma separated>"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
from floor_progress_notify import standready_eval
ck, n, pf = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); levels = [float(x) for x in sys.argv[4].split(",")]
st = "/home/ubuntu/workspace/standai/checkpoints/autosave/ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip"
for lv in levels:
    ev = standready_eval(ck, st, n=n, crouch_frac=0.0, max_steps=450, prone_frac=pf, down_timeout=False, level=lv)
    a, r, h = ev["lying"]
    print(f"{ck.split('/')[-1]} level {lv:.2f}: starts {a}, reached {r} ({r/a:.0%}), held by stand {h} ({h/a:.0%} of starts, {h/max(r,1):.0%} of reached)", flush=True)
