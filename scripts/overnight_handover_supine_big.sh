#!/bin/bash
# Supine hand-over policy s3: arrivals of the big supine floor policy (fdist, level 0.3), warm start from the big stand policy.
cd /home/ubuntu/workspace/standai/scripts
while pgrep -f overnight_handover_big.sh >/dev/null; do sleep 20; done
../.venv/bin/python - <<'PY'
import numpy as np, glob
ds = [np.load(f, allow_pickle=True) for f in sorted(glob.glob("../checkpoints/pool_fd_supine_?.npz"))]
out = {k: np.concatenate([d[k] for d in ds]) for k in ("qpos", "qvel", "head")}
np.savez("../checkpoints/pool_fd_supine_all.npz", **out)
print({k: v.shape for k, v in out.items()})
PY
export STANDAI_NO_RECOVERY_GATE=1   # training only; evals below run with env -u
nohup ../.venv/bin/python handover_progress_notify.py --log ../checkpoints/handover_train_s3.log --ckpt-glob '../checkpoints/autosave/ppo_handover_s3_*_steps.zip' --side supine --interval-minutes 30 >> ../checkpoints/handover_progress_notify_s3.log 2>&1 &
NP=$!
../.venv/bin/python train_handover.py --n-envs 8 --timesteps 30000000 --out ppo_handover_s3 --pool ../checkpoints/pool_fd_supine_all.npz --first-index 0 --ladder-start 1.0 --target-kl 0.03 --learning-rate 1e-4 --std-min 0.3 --init-from ../checkpoints/best_stand_big1.zip >> ../checkpoints/handover_train_s3.log 2>&1
kill $NP
CK=$(ls -t ../checkpoints/autosave/ppo_handover_s3_*_steps.zip | head -1)
cp "$CK" ../checkpoints/handover_supine_s3.zip
env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/controller_side_eval.py --handover none ../checkpoints/handover_supine_s2.zip ../checkpoints/handover_supine_s3.zip --sides supine --levels 0.7 --n 200 --seed0 22000 --supine ../checkpoints/best_floor_supine_fdist.zip --prone ../checkpoints/best_prone8.zip --stand ../checkpoints/best_stand_big1.zip --workers 8 > ../checkpoints/analysis/ctl_supine_s3.out 2>&1
../.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from kakao_notify import send_message
txt = open("../checkpoints/analysis/ctl_supine_s3.out").read()
lines = [l[:120] for l in txt.splitlines() if "standing at" in l]
send_message("[s3 done] supine 20 s standing (seeds 22000+, 200)\n" + "\n".join(lines))
PY
