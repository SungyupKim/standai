#!/bin/bash
# Larger-network PRONE floor policy, after the supine pipeline (overnight_big.sh) has finished: stage 1 (loose target, from scratch) -> check -> stage 2 (staged stand-ready target) ->
# whole-system prone eval vs prone8 (with / without the prone hand-over policy) -> Kakao.
cd /home/ubuntu/workspace/standai/scripts
export STANDAI_NO_RECOVERY_GATE=1
while kill -0 792653 2>/dev/null; do sleep 60; done
nohup ../.venv/bin/python floor_progress_notify.py --log ../checkpoints/floor_train_bigp1.log --ckpt-glob '../checkpoints/autosave/ppo_floor_bigp1_*_steps.zip' --total-timesteps 10000000 --interval-minutes 60 --max-steps 450 --prone-frac 1.0 --no-down-timeout --eval-n 100 >> ../checkpoints/floor_progress_notify_bigp1.log 2>&1 &
../.venv/bin/python train_floor.py --n-envs 8 --timesteps 10000000 --out ppo_floor_bigp1 --prone-frac 1.0 --max-steps 450 --no-down-timeout --floor-burst-rate 0.0 --net-arch 256 256 --batch-size 256 >> ../checkpoints/floor_train_bigp1.log 2>&1
../.venv/bin/python - <<'PY'
import sys, re; sys.path.insert(0, ".")
from kakao_notify import send_message
ts=None; last=0
for l in open("../checkpoints/floor_train_bigp1.log", errors="ignore"):
    m=re.match(r"\|\s+total_timesteps\s+\|\s+(\d+)", l)
    if m: ts=int(m.group(1))
    m=re.match(r"\|\s+success_rate\s+\|\s+([\d.]+)", l)
    if m: last=float(m.group(1))
open("../checkpoints/bigp1_last_success.txt","w").write(f"{last}\n")
send_message(f"[bigp1 stage 1 done] prone, 256x256, loose target: final training success {last:.2f}" + (" -> stage 2 starts" if last>=0.2 else " -> < 0.2: stage 2 skipped"))
PY
LAST=$(cat ../checkpoints/bigp1_last_success.txt)
if [ "$(echo "$LAST < 0.2" | bc -l)" = "1" ]; then exit 0; fi
B1=$(ls -t ../checkpoints/autosave/ppo_floor_bigp1_*_steps.zip | head -1); cp "$B1" ../checkpoints/floor_prone_bigp1.zip
nohup ../.venv/bin/python floor_progress_notify.py --log ../checkpoints/floor_train_bigp2.log --ckpt-glob '../checkpoints/autosave/ppo_floor_bigp2_*_steps.zip' --total-timesteps 25000000 --interval-minutes 120 --standready --crouch-frac 0.0 --max-steps 450 --prone-frac 1.0 --no-down-timeout --stand-ckpt ../checkpoints/best_stand_v55b.zip --eval-n 100 >> ../checkpoints/floor_progress_notify_bigp2.log 2>&1 &
../.venv/bin/python train_floor.py --n-envs 8 --timesteps 25000000 --out ppo_floor_bigp2 --standready --crouch-frac 0.0 --prone-frac 1.0 --max-steps 450 --no-down-timeout --speed-pen 0.0 --air-pen 0.0 --level-start 0.0 --level-gate 0.35 --level-step 0.05 --learning-rate 1e-4 --target-kl 0.03 --floor-burst-rate 0.0 --init-from ../checkpoints/floor_prone_bigp1.zip >> ../checkpoints/floor_train_bigp2.log 2>&1
B2=$(ls -t ../checkpoints/autosave/ppo_floor_bigp2_*_steps.zip | head -1); cp "$B2" ../checkpoints/floor_prone_bigp2.zip
OUT=../checkpoints/analysis/ctl_prone_bigp2.out; : > $OUT
for FLOOR in ../checkpoints/best_prone8.zip ../checkpoints/floor_prone_bigp2.zip; do
  ../.venv/bin/python -I analysis/controller_side_eval.py --prone $FLOOR --handover none ../checkpoints/best_handover_prone.zip --sides prone --levels 0.7 --n 200 --seed0 29000 --stand ../checkpoints/best_stand_v55b.zip --workers 8 | sed "s|^|prone floor $(basename $FLOOR | cut -c1-24) | " >> $OUT
done
LEVEL=$(ls -t ../checkpoints/*_level.txt 2>/dev/null | head -1 | xargs cat 2>/dev/null)
../.venv/bin/python - "$LEVEL" <<'PY'
import sys; sys.path.insert(0, ".")
from kakao_notify import send_message
lines=[l.replace("hand-over ","").replace("prone  level ","L")[:150] for l in open("../checkpoints/analysis/ctl_prone_bigp2.out").read().splitlines() if "standing at" in l]
send_message(f"[bigp2 done] stage-2 level {sys.argv[1]}; prone 20 s standing, seeds 29000+, 200 each (prone8 vs larger network; hand-over none / best_handover_prone)\n"+"\n".join(lines))
PY
