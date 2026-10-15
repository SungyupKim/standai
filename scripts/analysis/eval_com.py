import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
from floor_progress_notify import standready_eval
ck, com, n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
st = "/home/ubuntu/workspace/standai/checkpoints/autosave/ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip"
ev = standready_eval(ck, st, n=n, crouch_frac=0.0, max_steps=450, prone_frac=1.0, down_timeout=False, level=1.0, com_max=com)
a, r, h = ev["lying"]
print(f"com_max {com:.2f} m: prone starts {a}, reached {r} ({r/a:.0%}), held by stand {h} ({h/a:.0%} of starts, {h/max(r,1):.0%} of reached)", flush=True)
