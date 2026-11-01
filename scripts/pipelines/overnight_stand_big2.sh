#!/bin/bash
# Stand policy big2: continue best_stand_big1 on hand-over states of the NEW supine floor policy (fdist2, level 0.7; stand_big1 catches only 24% of them vs 69% of the v12 arrivals it was trained on)
# plus the unchanged prone arrivals, then compare in the controller (hand-over policy = h4, supine = fdist2 best, prone = prone8) on fresh seeds. No `export` of training variables.
cd /home/ubuntu/workspace/standai/scripts
K() { ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from kakao_notify import send_message; send_message(sys.stdin.read())"; }
while [ "$(ls ../checkpoints/pool_fd2_supine_?.npz 2>/dev/null | wc -l)" -lt 4 ]; do sleep 20; done
sleep 5
../.venv/bin/python - <<'PY'
import numpy as np, glob
ds = [np.load(f, allow_pickle=True) for f in sorted(glob.glob("../checkpoints/pool_fd2_supine_?.npz"))]
b = np.load("../checkpoints/pool_big_all.npz", allow_pickle=True)
out = {k: np.concatenate([d[k] for d in ds] + [b[k][16000:]]) for k in ("qpos", "qvel", "head")}
np.savez("../checkpoints/pool_fd2_all.npz", **out)
print({k: v.shape for k, v in out.items()})
PY
P=../checkpoints/pool_fd2_all.npz
TRAINENV="STANDAI_NO_RECOVERY_GATE=1 STANDAI_HARSH_MIN_TORSO=0.7 STANDAI_START_VEL_POOL=$P STANDAI_START_VEL_PROB=0.6 STANDAI_START_VEL_JOINT=3.5 STANDAI_MIX_POOL=$P STANDAI_MIX_PROB=0.3 STANDAI_MIX_LAM_LO=0.7 STANDAI_STEP_K=300"
echo "[stand big2] pool ready, training starts" | K
env $TRAINENV ../.venv/bin/python train_locomotion.py --n-envs 8 --timesteps 30000000 --stand-only --out ppo_stand_big2 --height-start-scale 15.0 --height-end-scale 15.0 --height-already-relaxed --curriculum-fall-gate 0.65 --curriculum-streak-gate 350 --calm-already-relaxed --calm-end-frac 0.5 --std-max 0.4 --no-knee-floor --learning-rate 5e-5 --target-kl 0.03 --handoff-pool $P --handoff-frac 0.4 --com-ladder --ladder-first-index 16000 --ladder-gate 0.25 --init-from ../checkpoints/best_stand_big1.zip >> ../checkpoints/locomotion_train_stand_big2.log 2>&1
L=$(ls -t ../checkpoints/autosave/ppo_stand_big2_*_steps.zip | head -1); cp "$L" ../checkpoints/stand_big2_last.zip
OUT=../checkpoints/analysis/ctl_stand_big2.out; : > $OUT
for SEED in 55000 56000; do
  for ST in ../checkpoints/best_stand_big1.zip ../checkpoints/stand_big2_last.zip; do
    env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/controller_side_eval.py --handover ../checkpoints/best_handover_prone.zip --sides supine prone --levels 0.7 --n 200 --seed0 $SEED --supine ../checkpoints/best_floor_supine_fdist.zip --prone ../checkpoints/best_prone8.zip --stand $ST --workers 8 | sed "s|^|seeds $SEED stand $(basename $ST) |" >> $OUT
  done
done
env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/catch_by_com.py ../checkpoints/best_stand_big1.zip ../checkpoints/stand_big2_last.zip --sets eval2_all eval_fd2_all > ../checkpoints/analysis/catch_stand_big2.out 2>&1
{ echo "[stand big2 done] controller 20 s standing, 200 per cell, fresh seeds"; grep "standing at" $OUT | sed -E 's/hand-over [^ ]+ +//; s/level 0.7 v_gate None flip 0: //; s/seeds ([0-9]+) stand ([^ ]+)/\1 \2/' | cut -c1-110; grep -E "^(==|supine: all|prone: all)" ../checkpoints/analysis/catch_stand_big2.out; } | K
