#!/bin/bash
# catch2: continue the catch1 run (15M checkpoint) for 25M more steps with a faster catch rollout (STANDAI_CATCH_EARLY=90: held 1.8 s -> caught, stop; labels differ from the full 3 s rollout on ~7% of arrivals),
# then compare arr1 / catch1 15M / catch2 in the controller on FOUR fresh seed sets (800 starts per arm).
# No `export`: training variables are given with `env` to the training command only.
cd /home/ubuntu/workspace/standai/scripts
K() { ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from kakao_notify import send_message; send_message(sys.stdin.read())"; }
nohup nice -n 10 ../.venv/bin/python floor_trend_notify.py --glob '../checkpoints/autosave/ppo_floor_catch2_*_steps.zip' --label catch2 --total-steps 208000000 --arr-k 15 --arr-v 0.45 --baseline-ckpt ../checkpoints/floor_supine_catch1_15m.zip --baseline-label 'catch1 15M' > ../checkpoints/trend_catch2.log 2>&1 &
NP=$!
env STANDAI_NO_RECOVERY_GATE=1 STANDAI_ARRIVAL_K=15 STANDAI_CATCH_K=20 STANDAI_CATCH_EARLY=90 STANDAI_CATCH_STAND=../checkpoints/best_stand_big1.zip ../.venv/bin/python train_floor.py --n-envs 8 --timesteps 25000000 --out ppo_floor_catch2 --standready --crouch-frac 0.0 --supine-frac 1.0 --max-steps 450 --no-down-timeout --floor-burst-rate 0.0 --speed-pen 0.08 --speed-vz-w 0.25 --level-start 0.7 --level-gate 2.0 --learning-rate 5e-5 --target-kl 0.03 --init-from ../checkpoints/floor_supine_catch1_15m.zip >> ../checkpoints/floor_train_catch2.log 2>&1
kill $NP 2>/dev/null
L=$(ls -t ../checkpoints/autosave/ppo_floor_catch2_*_steps.zip | head -1); cp "$L" ../checkpoints/floor_supine_catch2_last.zip
OUT=../checkpoints/analysis/ctl_supine_catch2.out; : > $OUT
for SEED in 68000 69000 70000 71000; do
  for FLOOR in ../checkpoints/floor_supine_arr1_last.zip ../checkpoints/floor_supine_catch1_15m.zip ../checkpoints/floor_supine_catch2_last.zip; do
    env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none --sides supine --levels 0.7 --n 200 --seed0 $SEED --stand ../checkpoints/best_stand_big1.zip --workers 8 | sed "s|^|seeds $SEED floor $(basename $FLOOR | cut -c1-32) |" >> $OUT
  done
done
{ echo "[catch2 done] 서기 결과 보상 +25M(롤아웃 단축); 누운쪽 20초 서기, 세트당 200개 x 4세트"; grep "standing at" $OUT | sed -E 's/hand-over none +//; s/supine level 0.7 v_gate None flip 0: //; s/seeds ([0-9]+) floor ([^ ]+)/\1 \2/; s/\(10 s [0-9]+, handed over [0-9]+\)//' | cut -c1-100; } | K
