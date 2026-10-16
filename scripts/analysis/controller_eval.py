"""End-to-end 'does the body end up standing?' for the rule-based GetUpController (scripts/controller.py), natural start mix, fixed seeds.
Standing = torso >= 0.95 m and |pitch| < 30 deg for the last 25 steps (same definition as system_loop.py). Reports standing at 10 s and at the end,
hand-overs, stand-policy falls. Usage: controller_eval.py <trigger_level> <fall_back 0|1> <n_seeds> [steps=1000] [prone_ckpt] [supine_ckpt] [stand_ckpt]"""
import sys
sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np
from controller import GetUpController
from floor_env import FloorEnv

A = "/home/ubuntu/workspace/standai/checkpoints/"
level, back, n = float(sys.argv[1]), bool(int(sys.argv[2])), int(sys.argv[3])
T = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
prone = sys.argv[5] if len(sys.argv) > 5 else A + "best_prone8.zip"
supine = sys.argv[6] if len(sys.argv) > 6 else A + "autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip"
stand = sys.argv[7] if len(sys.argv) > 7 else A + "autosave/ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip"
env = FloorEnv(standready=True, level=level, crouch_frac=0.0, max_steps=10 ** 6, prone_frac=0.0, down_timeout=False)
ctl = GetUpController(supine, prone, stand, fall_back=back)
res = dict(s10=0, send=0, hand=0, falls=0)
by_side = {"supine": [0, 0], "prone": [0, 0]}
for seed in range(n):
    obs, _ = env.reset(seed=seed)
    ctl.reset(env)
    side = ctl.floor_name
    hist = []
    for t in range(T):
        obs, _, term, trunc, info = env.step(ctl.act(obs, env))
        ctl.observe(env, info)
        hist.append((float(env.data.xpos[env.a_torso_id][2]), ctl.pitch(env)))
    ok = lambda lo, hi: all(z >= 0.95 and abs(np.degrees(p)) < 30 for z, p in hist[lo:hi])
    s10, send = ok(500 - 25, 500), ok(T - 25, T)
    res["s10"] += s10; res["send"] += send; res["hand"] += ctl.handovers > 0; res["falls"] += ctl.falls
    by_side[side][0] += 1; by_side[side][1] += send
print(f"trigger level {level:.2f}, fall back {back}: {n} starts, {T * 0.02:.0f} s | standing at 10 s {res['s10']} ({res['s10'] / n:.0%}) | standing at the end "
      f"{res['send']} ({res['send'] / n:.0%}) | handed over at least once {res['hand']} ({res['hand'] / n:.0%}) | stand falls {res['falls']} | "
      f"end-standing by start side: supine {by_side['supine'][1]}/{by_side['supine'][0]}, prone {by_side['prone'][1]}/{by_side['prone'][0]}", flush=True)
