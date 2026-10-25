#!/bin/bash
# Learned hand-over gate sweep (diagnostic): supine side, big1, no hand-over policy, 3 fresh seed sets of 200; gate off / catchability >= 0.3 / 0.5 / 0.7 (wait <= 50 steps). Variables are given with `env` per command, never exported.
cd /home/ubuntu/workspace/standai
OUT=checkpoints/analysis/clf_gate.out; : > $OUT
for SEED in 73000 74000 75000; do
  for G in off 0.3 0.5 0.7; do
    if [ "$G" = off ]; then E="X=1"; else E="STANDAI_CLF_GATE=$G"; fi
    env -u STANDAI_NO_RECOVERY_GATE $E .venv/bin/python -I scripts/analysis/controller_side_eval.py --handover none --sides supine --levels 0.7 --n 200 --seed0 $SEED --supine checkpoints/best_floor_supine_fdist.zip --prone checkpoints/best_prone8.zip --stand checkpoints/best_stand_big1.zip --workers 8 | sed "s|^|seeds $SEED gate=$G |" >> $OUT
  done
done
