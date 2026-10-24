#!/bin/bash
# Stops the catch1 run at ~15M steps (counter 183M; the run is ~3.5M steps/hour, 40M would take 11 h), checks that checkpoint against arr1 in the controller on three fresh seed sets (600 starts per arm), sends the result by Kakao.
# Usage: catch1_15m.sh <pid> <pid> ...   (explicit PIDs of overnight_floor_catch.sh, its train_floor.py and notifier; never pkill -f)
cd /home/ubuntu/workspace/standai/scripts
K() { ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from kakao_notify import send_message; send_message(sys.stdin.read())"; }
PICK='import glob,re,zipfile,os,time
fs=sorted(glob.glob("../checkpoints/autosave/ppo_floor_catch1_*_steps.zip"),key=lambda f:int(re.search(r"_(\d+)_steps",f).group(1)))
g=[f for f in fs if int(re.search(r"_(\d+)_steps",f).group(1))>=183000000 and os.path.getsize(f)>0 and zipfile.is_zipfile(f) and time.time()-os.path.getmtime(f)>60]
print(g[0] if g else "")'
until [ -n "$(../.venv/bin/python -c "$PICK")" ]; do sleep 60; done
CK=$(../.venv/bin/python -c "$PICK")
cp "$CK" ../checkpoints/floor_supine_catch1_15m.zip
kill $@ 2>/dev/null; sleep 3
for P in $(ps -eo pid,ppid,args | awk -v pids="$*" 'BEGIN{split(pids,a," "); for(i in a) s[a[i]]=1} ($2 in s) {print $1}'); do kill $P 2>/dev/null; done
OUT=../checkpoints/analysis/ctl_supine_catch1_15m.out; : > $OUT
for SEED in 68000 69000 70000; do
  for FLOOR in ../checkpoints/floor_supine_arr1_last.zip ../checkpoints/floor_supine_catch1_15m.zip; do
    env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none --sides supine --levels 0.7 --n 200 --seed0 $SEED --stand ../checkpoints/best_stand_big1.zip --workers 8 | sed "s|^|seeds $SEED floor $(basename $FLOOR | cut -c1-32) |" >> $OUT
  done
done
{ echo "[catch1 15M] 도착 후 서기 결과 보상, 15M 스텝 지점; 누운쪽 20초 서기, 세트당 200개 x 3세트"; grep "standing at" $OUT | sed -E 's/hand-over none +//; s/supine level 0.7 v_gate None flip 0: //; s/seeds ([0-9]+) floor ([^ ]+)/\1 \2/; s/\(10 s [0-9]+, handed over [0-9]+\)//' | cut -c1-100; } | K
