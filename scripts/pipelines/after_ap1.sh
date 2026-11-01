#!/bin/bash
# After the from-scratch supine floor run with the arm-push reward (ap1): whole-system supine standing on fresh seeds, ap1 vs v12, hand-over none / s2, loose-to-normal hand-over levels.
cd /home/ubuntu/workspace/standai/scripts
export STANDAI_NO_RECOVERY_GATE=1
until grep -q "saved model to" ../checkpoints/floor_train_ap1.log; do sleep 60; done
sleep 10
AP=$(ls -t ../checkpoints/autosave/ppo_floor_ap1_*_steps.zip | head -1); cp "$AP" ../checkpoints/floor_supine_ap1.zip
V12=../checkpoints/autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip
OUT=../checkpoints/analysis/ctl_supine_ap1.out; : > $OUT
for FLOOR in $V12 ../checkpoints/floor_supine_ap1.zip; do
  ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none ../checkpoints/handover_supine_s2.zip --sides supine --levels 0.3 0.5 0.7 --n 100 --seed0 21000 --stand ../checkpoints/best_stand_v55b.zip --workers 8 | sed "s|^|floor $(basename $FLOOR | cut -c1-24) | " >> $OUT
done
../.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from kakao_notify import send_message
lines = [l.replace("hand-over ", "").replace("supine level ", "L")[:150] for l in open("../checkpoints/analysis/ctl_supine_ap1.out").read().splitlines() if "standing at" in l]
send_message("[ap1 done] supine 20 s standing, seeds 21000+, 100 each (v12 vs arm-push floor policy from scratch)\n" + "\n".join(lines))
PY
