#!/bin/bash
# 1) whole system with the promoted supine floor policy (arr1 last) on fresh seeds, 2) arr2: stronger arrival reward (K 25, target speed 0.6 m/s) continued from arr1 last for 40M steps, 3) compare in the controller.
# No `export`: training variables are given with `env` to the training command only.
cd /home/ubuntu/workspace/standai/scripts
K() { ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from kakao_notify import send_message; send_message(sys.stdin.read())"; }
OUT=../checkpoints/analysis/ctl_full_arr1.out; : > $OUT
for SEED in 63000 64000; do
  env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/controller_side_eval.py --handover ../checkpoints/best_handover_prone.zip --sides supine prone --levels 0.7 --n 200 --seed0 $SEED --supine ../checkpoints/best_floor_supine_fdist.zip --prone ../checkpoints/best_prone8.zip --stand ../checkpoints/best_stand_big1.zip --workers 8 | sed "s|^|seeds $SEED |" >> $OUT
done
{ echo "[전체 시스템, 누운 바닥 = arr1] 20초 서기, 새 시드 200개씩"; grep "standing at" $OUT | sed -E 's/hand-over best_handover_prone.zip +//; s/level 0.7 v_gate None flip 0: //; s/\(10 s [0-9]+, handed over [0-9]+\)//' | cut -c1-90; } | K
nohup nice -n 10 ../.venv/bin/python floor_trend_notify.py --glob '../checkpoints/autosave/ppo_floor_arr2_*_steps.zip' --label arr2 --total-steps 208000000 --arr-k 25 --arr-v 0.6 --baseline-ckpt ../checkpoints/floor_supine_arr1_last.zip --baseline-label 'arr1 마지막' > ../checkpoints/trend_arr2.log 2>&1 &
NP=$!
env STANDAI_NO_RECOVERY_GATE=1 STANDAI_ARRIVAL_K=25 STANDAI_ARRIVAL_VX=0.6 ../.venv/bin/python train_floor.py --n-envs 8 --timesteps 40000000 --out ppo_floor_arr2 --standready --crouch-frac 0.0 --supine-frac 1.0 --max-steps 450 --no-down-timeout --floor-burst-rate 0.0 --speed-pen 0.08 --speed-vz-w 0.25 --level-start 0.7 --level-gate 2.0 --learning-rate 5e-5 --target-kl 0.03 --init-from ../checkpoints/floor_supine_arr1_last.zip >> ../checkpoints/floor_train_arr2.log 2>&1
kill $NP 2>/dev/null
L=$(ls -t ../checkpoints/autosave/ppo_floor_arr2_*_steps.zip | head -1); cp "$L" ../checkpoints/floor_supine_arr2_last.zip
OUT2=../checkpoints/analysis/ctl_supine_arr2.out; : > $OUT2
for SEED in 65000 66000; do
  for FLOOR in ../checkpoints/floor_supine_arr1_last.zip ../checkpoints/floor_supine_arr2_last.zip; do
    env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none --sides supine --levels 0.7 --n 200 --seed0 $SEED --stand ../checkpoints/best_stand_big1.zip --workers 8 | sed "s|^|seeds $SEED floor $(basename $FLOOR | cut -c1-32) |" >> $OUT2
  done
done
{ echo "[arr2 done] 보상 강화(K 25, 목표 0.6 m/s) +40M; 누운쪽 20초 서기, 200개씩"; grep "standing at" $OUT2 | sed -E 's/hand-over none +//; s/supine level 0.7 v_gate None flip 0: //; s/seeds ([0-9]+) floor ([^ ]+)/\1 \2/; s/\(10 s [0-9]+, handed over [0-9]+\)//' | cut -c1-100; } | K
