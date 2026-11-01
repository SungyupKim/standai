#!/bin/bash
# Larger-network supine floor policy, unattended: stage 1 (big1, running) -> check -> stage 2 (big2: staged stand-ready target, from big1) -> whole-system supine eval vs v12 -> Kakao.
cd /home/ubuntu/workspace/standai/scripts
export STANDAI_NO_RECOVERY_GATE=1
until grep -q "saved model to" ../checkpoints/floor_train_big1.log; do sleep 60; done
sleep 10
../.venv/bin/python - <<'PY'
import sys, re; sys.path.insert(0, ".")
from kakao_notify import send_message
def series(path):
    out=[]; ts=None
    for l in open(path, errors="ignore"):
        m=re.match(r"\|\s+total_timesteps\s+\|\s+(\d+)", l)
        if m: ts=int(m.group(1))
        m=re.match(r"\|\s+success_rate\s+\|\s+([\d.]+)", l)
        if m and ts: out.append((ts, float(m.group(1))))
    return out
def at(s, t): 
    c=[x for x in s if x[0]>=t]; return f"{c[0][1]:.2f}" if c else "-"
a=series("../checkpoints/floor_train_s4a.log"); b=series("../checkpoints/floor_train_big1.log")
last=b[-1][1] if b else 0
msg="[big1 stage 1 done] training success (log): " + " | ".join(f"{t/1e6:.0f}M big1 {at(b,t)} vs s4a {at(a,t)}" for t in (2e6,3e6,4e6,5e6,8e6)) + f" | final big1 {last:.2f}"
send_message(msg + ("\n-> stage 2 starts" if last>=0.2 else "\n-> success < 0.2: stage 2 skipped"))
open("../checkpoints/big1_last_success.txt","w").write(f"{last}\n")
PY
LAST=$(cat ../checkpoints/big1_last_success.txt)
if [ "$(echo "$LAST < 0.2" | bc -l)" = "1" ]; then exit 0; fi
B1=$(ls -t ../checkpoints/autosave/ppo_floor_big1_*_steps.zip | head -1)
cp "$B1" ../checkpoints/floor_supine_big1.zip
nohup ../.venv/bin/python floor_progress_notify.py --log ../checkpoints/floor_train_big2.log --ckpt-glob '../checkpoints/autosave/ppo_floor_big2_*_steps.zip' --total-timesteps 25000000 --interval-minutes 120 --standready --crouch-frac 0.0 --max-steps 450 --supine-frac 1.0 --no-down-timeout --stand-ckpt ../checkpoints/best_stand_v55b.zip --eval-n 100 >> ../checkpoints/floor_progress_notify_big2.log 2>&1 &
../.venv/bin/python train_floor.py --n-envs 8 --timesteps 25000000 --out ppo_floor_big2 --standready --crouch-frac 0.0 --supine-frac 1.0 --max-steps 450 --no-down-timeout --speed-pen 0.04 --speed-vz-w 0.25 --level-start 0.0 --level-gate 0.35 --level-step 0.05 --learning-rate 1e-4 --target-kl 0.03 --floor-burst-rate 0.0 --init-from ../checkpoints/floor_supine_big1.zip >> ../checkpoints/floor_train_big2.log 2>&1
B2=$(ls -t ../checkpoints/autosave/ppo_floor_big2_*_steps.zip | head -1); cp "$B2" ../checkpoints/floor_supine_big2.zip
V12=../checkpoints/autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip
OUT=../checkpoints/analysis/ctl_supine_big2.out; : > $OUT
for FLOOR in $V12 ../checkpoints/floor_supine_big2.zip; do
  ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none ../checkpoints/handover_supine_s2.zip --sides supine --levels 0.5 0.7 --n 200 --seed0 27000 --stand ../checkpoints/best_stand_v55b.zip --workers 8 | sed "s|^|floor $(basename $FLOOR | cut -c1-26) | " >> $OUT
done
LEVEL=$(ls -t ../checkpoints/*_level.txt 2>/dev/null | head -1 | xargs cat 2>/dev/null)
../.venv/bin/python - "$LEVEL" <<'PY'
import sys; sys.path.insert(0, ".")
from kakao_notify import send_message
lines=[l.replace("hand-over ","").replace("supine level ","L")[:150] for l in open("../checkpoints/analysis/ctl_supine_big2.out").read().splitlines() if "standing at" in l]
send_message(f"[big2 done] stage-2 level reached {sys.argv[1]}; supine 20 s standing, seeds 27000+, 200 each (v12 vs larger network)\n"+"\n".join(lines))
PY
