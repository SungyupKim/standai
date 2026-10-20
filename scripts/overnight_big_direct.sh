#!/bin/bash
# "All at once": larger-network SUPINE floor policy trained from scratch directly on the final target (stand-ready region, level 0.7 fixed, no loose stage, no curriculum), after the prone pipeline.
cd /home/ubuntu/workspace/standai/scripts
export STANDAI_NO_RECOVERY_GATE=1
while kill -0 803002 2>/dev/null; do sleep 60; done
nohup ../.venv/bin/python floor_progress_notify.py --log ../checkpoints/floor_train_bigd.log --ckpt-glob '../checkpoints/autosave/ppo_floor_bigd_*_steps.zip' --total-timesteps 20000000 --interval-minutes 60 --standready --crouch-frac 0.0 --max-steps 450 --supine-frac 1.0 --no-down-timeout --stand-ckpt ../checkpoints/best_stand_v55b.zip --eval-n 100 >> ../checkpoints/floor_progress_notify_bigd.log 2>&1 &
../.venv/bin/python train_floor.py --n-envs 8 --timesteps 20000000 --out ppo_floor_bigd --standready --crouch-frac 0.0 --supine-frac 1.0 --max-steps 450 --no-down-timeout --floor-burst-rate 0.0 --speed-pen 0.0 --level-start 0.7 --level-gate 2.0 --net-arch 256 256 --batch-size 256 >> ../checkpoints/floor_train_bigd.log 2>&1
B=$(ls -t ../checkpoints/autosave/ppo_floor_bigd_*_steps.zip | head -1); cp "$B" ../checkpoints/floor_supine_bigd.zip
V12=../checkpoints/autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip
OUT=../checkpoints/analysis/ctl_supine_bigd.out; : > $OUT
for FLOOR in $V12 ../checkpoints/floor_supine_bigd.zip; do
  ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none ../checkpoints/handover_supine_s2.zip --sides supine --levels 0.5 0.7 --n 200 --seed0 33000 --stand ../checkpoints/best_stand_v55b.zip --workers 8 | sed "s|^|floor $(basename $FLOOR | cut -c1-26) | " >> $OUT
done
../.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from kakao_notify import send_message
lines=[l.replace("hand-over ","").replace("supine level ","L")[:150] for l in open("../checkpoints/analysis/ctl_supine_bigd.out").read().splitlines() if "standing at" in l]
send_message("[bigd done] supine, 256x256 trained from scratch directly on the level-0.7 target (all at once); 20 s standing, seeds 33000+, 200 each\n"+"\n".join(lines))
PY
