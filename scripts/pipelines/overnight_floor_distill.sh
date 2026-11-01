#!/bin/bash
# Distil the supine floor policy v12 (64x64) into 256x256 (distill_floor.py), then PPO-fine-tune it on the v12 objective (level 0.7, supine, speed penalty 0.08/0.25) for 40M steps,
# scan the checkpoints, and compare with v12 on fresh seeds inside the controller (stand policy = the distilled+fine-tuned stand_big1_last), hand-over none / s2.
# NOTE: no `export` of training variables here (an export once leaked into the measurements).
cd /home/ubuntu/workspace/standai/scripts
export STANDAI_NO_RECOVERY_GATE=1
K() { ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from kakao_notify import send_message; send_message(sys.stdin.read())"; }
../.venv/bin/python distill_floor.py --out ../checkpoints/floor_distilled_256.zip >> ../checkpoints/distill_floor.log 2>&1
T=$(grep -E "^teacher +:" ../checkpoints/distill_floor.log | tail -1 | sed -E 's/.* in ([0-9]+)\/.*/\1/'); S=$(grep -E "^student +:" ../checkpoints/distill_floor.log | tail -1 | sed -E 's/.* in ([0-9]+)\/.*/\1/')
{ echo "[floor distillation done] v12 -> 256x256: hand-off region reached (level 0.7, 200 starts): teacher ${T:-?} / student ${S:-?}"; grep -E "round [0-9]|epoch +60" ../checkpoints/distill_floor.log | tail -3 | cut -c1-170; } | K
if [ -z "$T" ] || [ -z "$S" ] || [ "$(echo "$S < 0.6 * $T" | bc -l)" = "1" ]; then echo "[floor distillation] student reaches much less than the teacher (${S:-?} vs ${T:-?}): fine-tuning skipped" | K; exit 0; fi
../.venv/bin/python train_floor.py --n-envs 8 --timesteps 40000000 --out ppo_floor_fdist --standready --crouch-frac 0.0 --supine-frac 1.0 --max-steps 450 --no-down-timeout --floor-burst-rate 0.0 --speed-pen 0.08 --speed-vz-w 0.25 --level-start 0.7 --level-gate 2.0 --learning-rate 5e-5 --target-kl 0.03 --init-from ../checkpoints/floor_distilled_256.zip >> ../checkpoints/floor_train_fdist.log 2>&1
L=$(ls -t ../checkpoints/autosave/ppo_floor_fdist_*_steps.zip | head -1); cp "$L" ../checkpoints/floor_supine_fdist_last.zip
../.venv/bin/python -I analysis/floor_ckpt_scan.py --glob '../checkpoints/autosave/ppo_floor_fdist_*_steps.zip' --stride 40 --n 200 --seed0 26000 --out ../checkpoints/analysis/floor_ckpt_scan_fdist.json > ../checkpoints/analysis/floor_ckpt_scan_fdist.out 2>&1
BEST=$(../.venv/bin/python - <<'PY'
import json, numpy as np
d = json.load(open("../checkpoints/analysis/floor_ckpt_scan_fdist.json"))
print(max(d, key=lambda k: np.mean(d[k])))
PY
)
cp "$BEST" ../checkpoints/floor_supine_fdist_best.zip
V12=../checkpoints/autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip
OUT=../checkpoints/analysis/ctl_supine_fdist.out; : > $OUT
for FLOOR in $V12 ../checkpoints/floor_supine_fdist_last.zip ../checkpoints/floor_supine_fdist_best.zip; do
  ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none ../checkpoints/handover_supine_s2.zip --sides supine --levels 0.5 0.7 --n 200 --seed0 41000 --stand ../checkpoints/stand_big1_last.zip --workers 8 | sed "s|^|floor $(basename $FLOOR | cut -c1-30) | " >> $OUT
done
{ echo "[fdist done] supine floor policy distilled to 256x256 + 40M PPO steps; controller 20 s standing, seeds 41000+, 200 each, stand = stand_big1_last; scan: $(tail -2 ../checkpoints/analysis/floor_ckpt_scan_fdist.out | head -1 | cut -c1-110)"; grep "standing at" $OUT | sed -e 's/hand-over //' -e 's/supine level /L/' -e 's/v_gate None flip 0: //' | cut -c1-150; } | K
