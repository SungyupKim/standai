#!/bin/bash
# catch1: supine floor policy rewarded for the REAL outcome at arrival: at the success step the frozen stand policy (big1) is rolled out 3 s and STANDAI_CATCH_K is paid if it catches the state
# (plus the arr1 arrival score). Waits for the arr2 pipeline, continues from arr1 last for 40M steps, then compares arr1 vs catch1 in the controller on THREE fresh seed sets (600 starts per arm).
# No `export`: training variables are given with `env` to the training command only.
cd /home/ubuntu/workspace/standai/scripts
K() { ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from kakao_notify import send_message; send_message(sys.stdin.read())"; }
while pgrep -f overnight_arr2_resume.sh >/dev/null; do sleep 30; done
nohup nice -n 10 ../.venv/bin/python floor_trend_notify.py --glob '../checkpoints/autosave/ppo_floor_catch1_*_steps.zip' --label catch1 --total-steps 208000000 --arr-k 15 --arr-v 0.45 --baseline-ckpt ../checkpoints/floor_supine_arr1_last.zip --baseline-label 'arr1 마지막' > ../checkpoints/trend_catch1.log 2>&1 &
NP=$!
env STANDAI_NO_RECOVERY_GATE=1 STANDAI_ARRIVAL_K=15 STANDAI_CATCH_K=20 STANDAI_CATCH_STAND=../checkpoints/best_stand_big1.zip ../.venv/bin/python train_floor.py --n-envs 8 --timesteps 40000000 --out ppo_floor_catch1 --standready --crouch-frac 0.0 --supine-frac 1.0 --max-steps 450 --no-down-timeout --floor-burst-rate 0.0 --speed-pen 0.08 --speed-vz-w 0.25 --level-start 0.7 --level-gate 2.0 --learning-rate 5e-5 --target-kl 0.03 --init-from ../checkpoints/floor_supine_arr1_last.zip >> ../checkpoints/floor_train_catch1.log 2>&1
kill $NP 2>/dev/null
L=$(ls -t ../checkpoints/autosave/ppo_floor_catch1_*_steps.zip | head -1); cp "$L" ../checkpoints/floor_supine_catch1_last.zip
OUT=../checkpoints/analysis/ctl_supine_catch1.out; : > $OUT
for SEED in 68000 69000 70000; do
  for FLOOR in ../checkpoints/floor_supine_arr1_last.zip ../checkpoints/floor_supine_catch1_last.zip; do
    env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none --sides supine --levels 0.7 --n 200 --seed0 $SEED --stand ../checkpoints/best_stand_big1.zip --workers 8 | sed "s|^|seeds $SEED floor $(basename $FLOOR | cut -c1-32) |" >> $OUT
  done
done
{ echo "[catch1 done] 도착 후 실제 서기 결과 보상; 누운쪽 20초 서기, 세트당 200개 x 3세트"; grep "standing at" $OUT | sed -E 's/hand-over none +//; s/supine level 0.7 v_gate None flip 0: //; s/seeds ([0-9]+) floor ([^ ]+)/\1 \2/; s/\(10 s [0-9]+, handed over [0-9]+\)//' | cut -c1-100; } | K
