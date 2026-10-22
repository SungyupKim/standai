#!/bin/bash
# "Capacity ladder", unattended, after overnight_big_direct.sh (pid 830449) has measured the all-at-once larger-network supine floor policy (bigd).
#   A  bigd works (controller standing at level 0.7 >= 80/200 with or without s2): continue it 3 x 20M steps with a shrinking exploration-noise cap, scan the last phase's checkpoints, final eval vs v12.
#   B  it does not: distil the stand policy v55b into a 256x256 net (behaviour cloning + value regression), fine-tune it with PPO (hand-off pool + COM ladder, the v55b recipe) for 50M steps, then compare with v55b.
cd /home/ubuntu/workspace/standai/scripts
export STANDAI_NO_RECOVERY_GATE=1
K() { ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from kakao_notify import send_message; send_message(sys.stdin.read())"; }
while kill -0 830449 2>/dev/null; do sleep 60; done
BR=$(../.venv/bin/python - <<'PY'
import re
n_none = n_s2 = -1
try:
    for l in open("../checkpoints/analysis/ctl_supine_bigd.out"):
        if "floor_supine_bigd" in l and "level 0.7" in l and "standing at 20 s" in l:
            m = re.search(r"standing at 20 s (\d+)/200", l)
            if m:
                if "handover_supine_s2" in l: n_s2 = int(m.group(1))
                else: n_none = int(m.group(1))
except FileNotFoundError:
    pass
print("A" if max(n_none, n_s2) >= 80 else "B")
PY
)
echo "branch $BR" > ../checkpoints/overnight_ladder_branch.txt
echo "[ladder] bigd measured -> branch $BR ($( [ "$BR" = A ] && echo 'bigd works: continue it with a shrinking noise cap' || echo 'bigd did not work: distil + fine-tune the stand policy'))" | K

if [ "$BR" = "A" ]; then
  PREV=../checkpoints/floor_supine_bigd.zip; i=0
  for CFG in "0.40 0.20" "0.30 0.15" "0.20 0.10"; do
    set -- $CFG; i=$((i+1))
    ../.venv/bin/python train_floor.py --n-envs 8 --timesteps 20000000 --out ppo_floor_bigdA$i --standready --crouch-frac 0.0 --supine-frac 1.0 --max-steps 450 --no-down-timeout --floor-burst-rate 0.0 --speed-pen 0.0 --level-start 0.7 --level-gate 2.0 --learning-rate 1e-4 --target-kl 0.03 --std-max $1 --std-min $2 --init-from $PREV >> ../checkpoints/floor_train_bigdA.log 2>&1
    N=$(ls -t ../checkpoints/autosave/ppo_floor_bigdA${i}_*_steps.zip | head -1); cp "$N" ../checkpoints/floor_supine_bigdA$i.zip; PREV=../checkpoints/floor_supine_bigdA$i.zip
    LAST=$(grep -A4 "^| floor/ " ../checkpoints/floor_train_bigdA.log | grep success_rate | tail -1)
    echo "[ladder A] phase $i done (noise cap $1): last training success $LAST" | K
  done
  ../.venv/bin/python -I analysis/floor_ckpt_scan.py --glob '../checkpoints/autosave/ppo_floor_bigdA3_*_steps.zip' --stride 12 --n 200 --seed0 36000 --out ../checkpoints/analysis/floor_ckpt_scan_bigdA.json > ../checkpoints/analysis/floor_ckpt_scan_bigdA.out 2>&1
  BEST=$(../.venv/bin/python - <<'PY'
import json, numpy as np
d = json.load(open("../checkpoints/analysis/floor_ckpt_scan_bigdA.json"))
print(max(d, key=lambda k: np.mean(d[k])))
PY
)
  cp "$BEST" ../checkpoints/floor_supine_bigdA_best.zip
  V12=../checkpoints/autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip
  OUT=../checkpoints/analysis/ctl_supine_bigdA.out; : > $OUT
  for FLOOR in $V12 ../checkpoints/floor_supine_bigdA_best.zip; do
    ../.venv/bin/python -I analysis/controller_side_eval.py --supine $FLOOR --handover none ../checkpoints/handover_supine_s2.zip --sides supine --levels 0.5 0.7 --n 200 --seed0 37000 --stand ../checkpoints/best_stand_v55b.zip --workers 8 | sed "s|^|floor $(basename $FLOOR | cut -c1-26) | " >> $OUT
  done
  { echo "[ladder A done] supine, bigd + 60M steps with shrinking noise cap; scan top: $(tail -2 ../checkpoints/analysis/floor_ckpt_scan_bigdA.out | head -1 | cut -c1-120)"; grep "standing at" $OUT | sed -e 's/hand-over //' -e 's/supine level /L/' | cut -c1-150; } | K
else
  ../.venv/bin/python distill_stand.py --rounds 4 --steps-round0 250000 --steps-later 125000 --epochs 60 --out ../checkpoints/stand_distilled_256.zip >> ../checkpoints/distill_stand.log 2>&1
  { echo "[ladder B] distillation done:"; grep -E "^(teacher|student) +:|held-out" ../checkpoints/distill_stand.log | tail -4 | cut -c1-170; } | K
  # training-only start-distribution variables: passed with `env` to the trainer ONLY. (An `export` here once leaked into the final benchmark / funnel, which then measured near-standing starts.)
  TRAINENV="STANDAI_HARSH_MIN_TORSO=0.7 STANDAI_START_VEL_POOL=../checkpoints/pool_big_all.npz STANDAI_START_VEL_PROB=0.6 STANDAI_START_VEL_JOINT=3.5 STANDAI_MIX_POOL=../checkpoints/pool_big_all.npz STANDAI_MIX_PROB=0.3 STANDAI_MIX_LAM_LO=0.7 STANDAI_STEP_K=300"
  nohup ../.venv/bin/python stand_progress_notify.py --log ../checkpoints/locomotion_train_stand_big1.log --ckpt-glob '../checkpoints/autosave/ppo_stand_big1_*_steps.zip' --total-timesteps 50000000 --interval-minutes 120 --handoff-eval ../checkpoints/eval2_all.npz >> ../checkpoints/stand_progress_notify_big1.log 2>&1 &
  env $TRAINENV ../.venv/bin/python train_locomotion.py --n-envs 8 --timesteps 50000000 --stand-only --out ppo_stand_big1 --height-start-scale 15.0 --height-end-scale 15.0 --height-already-relaxed --curriculum-fall-gate 0.65 --curriculum-streak-gate 350 --calm-already-relaxed --calm-end-frac 0.5 --std-max 0.4 --no-knee-floor --learning-rate 5e-5 --target-kl 0.03 --handoff-pool ../checkpoints/pool_big_all.npz --handoff-frac 0.4 --com-ladder --ladder-first-index 16000 --ladder-gate 0.25 --init-from ../checkpoints/stand_distilled_256.zip >> ../checkpoints/locomotion_train_stand_big1.log 2>&1
  N=$(ls -t ../checkpoints/autosave/ppo_stand_big1_*_steps.zip | head -1); cp "$N" ../checkpoints/stand_big1_last.zip
  OUT=../checkpoints/analysis/ladder_B_final.out; : > $OUT
  ../.venv/bin/python -I analysis/benchmark.py --stand ../checkpoints/stand_big1_last.zip --tag big1 >> $OUT 2>&1
  for ST in ../checkpoints/best_stand_v55b.zip ../checkpoints/stand_big1_last.zip; do
    echo "== funnel with stand $(basename $ST)" >> $OUT
    ../.venv/bin/python -I analysis/funnel.py --stand $ST --ho-prone ../checkpoints/best_handover_prone.zip --n 200 --seed0 39000 >> $OUT 2>&1
  done
  { echo "[ladder B done] stand policy distilled to 256x256 + 50M PPO steps (vs v55b):"; grep -E "stand catch|controller, standing at 20 s|upright starts: gate|== funnel|end state|stood at some time" $OUT | cut -c1-170; } | K
fi
