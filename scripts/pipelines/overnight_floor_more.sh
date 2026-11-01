#!/bin/bash
# Continue the supine floor policy fdist for another 100M PPO steps (same objective), scan the checkpoints, and compare with fdist (current best) in the controller on two fresh seed sets.
# NOTE: no `export` of training variables here; evals run with the variable unset.
cd /home/ubuntu/workspace/standai/scripts
K() { ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from kakao_notify import send_message; send_message(sys.stdin.read())"; }
STANDAI_NO_RECOVERY_GATE=1 ../.venv/bin/python train_floor.py --n-envs 8 --timesteps 100000000 --out ppo_floor_fdist2 --standready --crouch-frac 0.0 --supine-frac 1.0 --max-steps 450 --no-down-timeout --floor-burst-rate 0.0 --speed-pen 0.08 --speed-vz-w 0.25 --level-start 0.7 --level-gate 2.0 --learning-rate 5e-5 --target-kl 0.03 --init-from ../checkpoints/floor_supine_fdist_last.zip >> ../checkpoints/floor_train_fdist2.log 2>&1
L=$(ls -t ../checkpoints/autosave/ppo_floor_fdist2_*_steps.zip | head -1); cp "$L" ../checkpoints/floor_supine_fdist2_last.zip
STANDAI_NO_RECOVERY_GATE=1 ../.venv/bin/python -I analysis/floor_ckpt_scan.py --glob '../checkpoints/autosave/ppo_floor_fdist2_*_steps.zip' --stride 40 --n 200 --seed0 27000 --out ../checkpoints/analysis/floor_ckpt_scan_fdist2.json > ../checkpoints/analysis/floor_ckpt_scan_fdist2.out 2>&1
BEST=$(../.venv/bin/python - <<'PY'
import json, numpy as np
d = json.load(open("../checkpoints/analysis/floor_ckpt_scan_fdist2.json"))
print(max(d, key=lambda k: np.mean(d[k])))
PY
)
cp "$BEST" ../checkpoints/floor_supine_fdist2_best.zip
OUT=../checkpoints/analysis/ctl_supine_fdist2.out; : > $OUT
for SEED in 51000 52000; do
  for FLOOR in ../checkpoints/best_floor_supine_fdist.zip ../checkpoints/floor_supine_fdist2_last.zip ../checkpoints/floor_supine_fdist2_best.zip; do
    env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none --sides supine --levels 0.7 --n 200 --seed0 $SEED --stand ../checkpoints/best_stand_big1.zip --workers 8 | sed "s|^|seeds $SEED floor $(basename $FLOOR | cut -c1-32) | " >> $OUT
  done
done
{ echo "[fdist2 done] supine floor +100M steps; controller 20 s standing, 200 per seed set, stand = best_stand_big1; scan: $(tail -2 ../checkpoints/analysis/floor_ckpt_scan_fdist2.out | head -1 | cut -c1-110)"; grep "standing at" $OUT | sed -e 's/hand-over none *//' -e 's/supine level 0.7 v_gate None flip 0: //' | cut -c1-150; } | K
