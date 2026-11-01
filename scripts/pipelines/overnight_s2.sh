#!/bin/bash
# Supine hand-over policy s2 on LOOSE-level arrivals: wait for the 4 pool parts -> merge -> train (warm start from the stand policy) -> controller eval -> Kakao message.
cd /home/ubuntu/workspace/standai/scripts
export STANDAI_NO_RECOVERY_GATE=1
while [ "$(ls ../checkpoints/pool_loose_supine_?.npz 2>/dev/null | wc -l)" -lt 4 ]; do sleep 20; done
sleep 5
../.venv/bin/python - <<'PY'
import numpy as np, glob
fs = sorted(glob.glob("../checkpoints/pool_loose_supine_?.npz"))
ds = [np.load(f, allow_pickle=True) for f in fs]
out = {k: np.concatenate([d[k] for d in ds]) for k in ds[0].files if ds[0][k].ndim >= 1}
np.savez("../checkpoints/pool_loose_supine_all.npz", **out)
print({k: v.shape for k, v in out.items()})
PY
nohup ../.venv/bin/python handover_progress_notify.py --log ../checkpoints/handover_train_s2.log --ckpt-glob '../checkpoints/autosave/ppo_handover_s2_*_steps.zip' --side supine --interval-minutes 90 >> ../checkpoints/handover_progress_notify_s2.log 2>&1 &
../.venv/bin/python train_handover.py --n-envs 8 --timesteps 12000000 --out ppo_handover_s2 --pool ../checkpoints/pool_loose_supine_all.npz --first-index 0 --ladder-start 1.0 --target-kl 0.03 --learning-rate 1e-4 --std-min 0.3 --init-from ../checkpoints/best_stand_v55b.zip >> ../checkpoints/handover_train_s2.log 2>&1
CK=$(ls -t ../checkpoints/autosave/ppo_handover_s2_*_steps.zip | head -1)
cp "$CK" ../checkpoints/handover_supine_s2.zip
../.venv/bin/python -I analysis/controller_side_eval.py --handover none ../checkpoints/handover_supine_s2.zip --sides supine --levels 0.3 0.5 0.7 --n 100 --seed0 9000 --stand ../checkpoints/best_stand_v55b.zip --workers 8 > ../checkpoints/analysis/ctl_supine_s2.out 2>&1
../.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from kakao_notify import send_message
txt = open("../checkpoints/analysis/ctl_supine_s2.out").read()
lines = [l.replace("hand-over ", "")[:120] for l in txt.splitlines() if "standing at" in l]
send_message("[s2 done] supine 20 s standing (seeds 9000+, ref: none 34/39/48 at level 0.3/0.5/0.7)\n" + "\n".join(lines))
PY
