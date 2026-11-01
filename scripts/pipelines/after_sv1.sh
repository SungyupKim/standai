#!/bin/bash
# Waits for the supine floor fine-tune sv1 to finish, then measures the supine side on fresh seeds (v12 vs sv1, with and without the supine hand-over policy s2) and sends the table by Kakao.
cd /home/ubuntu/workspace/standai/scripts
export STANDAI_NO_RECOVERY_GATE=1
until grep -q "saved model to" ../checkpoints/floor_train_sv1.log; do sleep 60; done
sleep 10
SV1=$(ls -t ../checkpoints/autosave/ppo_floor_sv1_*_steps.zip | head -1)
cp "$SV1" ../checkpoints/floor_supine_sv1.zip
V12=../checkpoints/autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip
OUT=../checkpoints/analysis/ctl_supine_sv1.out; : > $OUT
for FLOOR in $V12 ../checkpoints/floor_supine_sv1.zip; do
  ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none ../checkpoints/handover_supine_s2.zip --sides supine --levels 0.7 --n 200 --seed0 15000 --stand ../checkpoints/best_stand_v55b.zip --workers 8 | sed "s|^|floor $(basename $FLOOR | cut -c1-28) | " >> $OUT
done
../.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from kakao_notify import send_message
lines = [l[:170] for l in open("../checkpoints/analysis/ctl_supine_sv1.out").read().splitlines() if "standing at" in l]
send_message("[sv1 done] supine 20 s standing, 200 seeds 15000+ (v12 vs sv1 floor policy, hand-over none / s2)\n" + "\n".join(lines))
PY
