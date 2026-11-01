#!/bin/bash
# After the end-to-end run e2e1 (512x512) finishes: (1) paired final measurement of e2e1 vs the rule-based controller on three fresh seed sets (strict criteria), (2) network-size ablation:
# for 256x256, 128x128, 64x64: DAgger distillation of the controller, then 12M end-to-end PPO steps, (3) equal-step comparison (e2e1 @12M vs the three) on fresh seeds, Kakao message after each stage.
# No `export`: nothing is exported; the scripts clean STANDAI_* variables themselves.
cd /home/ubuntu/workspace/standai/scripts
K() { ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from kakao_notify import send_message; send_message(sys.stdin.read())"; }
while pgrep -f "train_getup.py --init-from ../checkpoints/getup_student_distilled.zip --out e2e1" >/dev/null; do sleep 60; done
FIN=../checkpoints/getup_e2e1_final.zip
[ -f $FIN ] || FIN=$(ls -t ../checkpoints/autosave/ppo_getup_e2e1_*_steps.zip | sed -n 2p)
OUT=../checkpoints/analysis/e2e1_final.out; : > $OUT
for S in 99100 99200 99300; do env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/getup_strict_eval.py --arms teacher $FIN --seed0 $S --n 200 --workers 8 | sed "s|^|seeds $S |" >> $OUT; done
{ echo "[e2e1 최종] 단일 네트워크 512x512 끝에서 끝까지 학습 완료; 새 시드 3세트(각 200/side), 엄격 기준"; grep -E "supine|prone" $OUT | sed -E 's/getup_e2e1_final.zip +/e2e1 /; s/teacher +/규칙 /; s/ \| standing fraction.*//' | cut -c1-120; } | K
for N in 256 128 64; do
  env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python distill_controller.py --net-arch $N $N --rounds 7 --out ../checkpoints/getup_student_$N.zip > ../checkpoints/distill_controller_$N.log 2>&1
  { echo "[크기 비교 $N x $N] 증류 완료"; grep "round 6" ../checkpoints/distill_controller_$N.log | sed -E 's/.*student/학생/; s/\(teacher.*//' | cut -c1-120; } | K
  nohup nice -n 10 ../.venv/bin/python getup_trend_notify.py --glob "../checkpoints/autosave/ppo_getup_e2e_${N}_*_steps.zip" --label e2e_$N --total-steps 12000000 > ../checkpoints/trend_e2e_$N.log 2>&1 &
  NP=$!
  env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python train_getup.py --init-from ../checkpoints/getup_student_$N.zip --out e2e_$N --timesteps 12000000 >> ../checkpoints/getup_train_e2e_$N.log 2>&1
  kill $NP 2>/dev/null
done
OUT2=../checkpoints/analysis/size_ablation.out; : > $OUT2
ARMS="teacher ../checkpoints/autosave/ppo_getup_e2e1_12000000_steps.zip"
for N in 256 128 64; do ARMS="$ARMS ../checkpoints/getup_e2e_${N}_final.zip"; done
env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/getup_strict_eval.py --arms $ARMS --seed0 99500 --n 200 --workers 8 >> $OUT2 2>&1
{ echo "[크기 비교] 같은 12M 스텝 끝에서 끝까지 학습, 새 시드 99500~(200/side), 엄격 기준"; grep -E "supine|prone" $OUT2 | sed -E 's/getup_e2e_([0-9]+)_final.zip +/\1x\1 /; s/ppo_getup_e2e1_12000000_steps.zip +/512x512 /; s/teacher +/규칙 /; s/ \| standing fraction.*//' | cut -c1-120; } | K
