#!/bin/bash
# Confirmation of the learned hand-over gate on three NEW seed sets (thresholds fixed beforehand: off / 0.3 / 0.2, wait <= 50 steps). Variables are given with `env` per command, never exported.
cd /home/ubuntu/workspace/standai
OUT=checkpoints/analysis/clf_gate_confirm.out; : > $OUT
for SEED in 76000 77000 78000; do
  for G in off 0.3 0.2; do
    if [ "$G" = off ]; then E="X=1"; else E="STANDAI_CLF_GATE=$G"; fi
    env -u STANDAI_NO_RECOVERY_GATE $E .venv/bin/python -I scripts/analysis/controller_side_eval.py --handover none --sides supine --levels 0.7 --n 200 --seed0 $SEED --supine checkpoints/best_floor_supine_fdist.zip --prone checkpoints/best_prone8.zip --stand checkpoints/best_stand_big1.zip --workers 8 | sed "s|^|seeds $SEED gate=$G |" >> $OUT
  done
done
