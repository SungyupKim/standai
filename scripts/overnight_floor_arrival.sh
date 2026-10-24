#!/bin/bash
# Supine floor policy with a continuous arrival reward (COM forward speed + pitch at the success step, STANDAI_ARRIVAL_K): continue best_floor_supine_fdist for 40M steps, scan, compare in the controller.
# No `export`: the training variables are passed with `env` to the training command only.
cd /home/ubuntu/workspace/standai/scripts
K() { ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from kakao_notify import send_message; send_message(sys.stdin.read())"; }
nohup nice -n 10 ../.venv/bin/python floor_trend_notify.py --glob '../checkpoints/autosave/ppo_floor_arr1_*_steps.zip' --label arr1 --total-steps 168000000 --ref 83 --ref-label 'fdist2 (시작점) 도달률' > ../checkpoints/trend_arr1.log 2>&1 &
NP=$!
env STANDAI_NO_RECOVERY_GATE=1 STANDAI_ARRIVAL_K=15 ../.venv/bin/python train_floor.py --n-envs 8 --timesteps 40000000 --out ppo_floor_arr1 --standready --crouch-frac 0.0 --supine-frac 1.0 --max-steps 450 --no-down-timeout --floor-burst-rate 0.0 --speed-pen 0.08 --speed-vz-w 0.25 --level-start 0.7 --level-gate 2.0 --learning-rate 5e-5 --target-kl 0.03 --init-from ../checkpoints/best_floor_supine_fdist.zip >> ../checkpoints/floor_train_arr1.log 2>&1
kill $NP
L=$(ls -t ../checkpoints/autosave/ppo_floor_arr1_*_steps.zip | head -1); cp "$L" ../checkpoints/floor_supine_arr1_last.zip
env STANDAI_NO_RECOVERY_GATE=1 ../.venv/bin/python -I analysis/floor_ckpt_scan.py --glob '../checkpoints/autosave/ppo_floor_arr1_*_steps.zip' --stride 40 --n 200 --seed0 27000 --out ../checkpoints/analysis/floor_ckpt_scan_arr1.json > ../checkpoints/analysis/floor_ckpt_scan_arr1.out 2>&1
BEST=$(../.venv/bin/python - <<'PY'
import json, numpy as np
d = json.load(open("../checkpoints/analysis/floor_ckpt_scan_arr1.json"))
print(max(d, key=lambda k: np.mean(d[k])))
PY
)
cp "$BEST" ../checkpoints/floor_supine_arr1_best.zip
OUT=../checkpoints/analysis/ctl_supine_arr1.out; : > $OUT
for SEED in 60000 61000; do
  for FLOOR in ../checkpoints/best_floor_supine_fdist.zip ../checkpoints/floor_supine_arr1_last.zip ../checkpoints/floor_supine_arr1_best.zip; do
    env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none --sides supine --levels 0.7 --n 200 --seed0 $SEED --stand ../checkpoints/best_stand_big1.zip --workers 8 | sed "s|^|seeds $SEED floor $(basename $FLOOR | cut -c1-32) |" >> $OUT
  done
done
env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/supine_fall_diag.py --n 400 --seed0 58000 --supine ../checkpoints/floor_supine_arr1_last.zip > ../checkpoints/analysis/supine_fall_diag_arr1.out 2>&1
{ echo "[arr1 done] supine floor + arrival reward (40M); controller 20 s standing, 200 per seed set"; grep "standing at" $OUT | sed -E 's/hand-over none +//; s/supine level 0.7 v_gate None flip 0: //; s/seeds ([0-9]+) floor ([^ ]+)/\1 \2/' | cut -c1-110; grep -E "^supine:|^com_vx|^pitch " ../checkpoints/analysis/supine_fall_diag_arr1.out; } | K
