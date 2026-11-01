#!/bin/bash
# Prone hand-over policy h4: warm start from the big (256x256) stand policy, then controller eval vs the old hand-over policy on fresh seeds.
cd /home/ubuntu/workspace/standai/scripts
export STANDAI_NO_RECOVERY_GATE=1   # training-only; the eval below runs with env -u
nohup ../.venv/bin/python handover_progress_notify.py --log ../checkpoints/handover_train_h4.log --ckpt-glob '../checkpoints/autosave/ppo_handover_h4_*_steps.zip' --side prone --interval-minutes 30 >> ../checkpoints/handover_progress_notify_h4.log 2>&1 &
NP=$!
../.venv/bin/python train_handover.py --n-envs 8 --timesteps 40000000 --out ppo_handover_h4 --pool ../checkpoints/pool_big_all.npz --first-index 16000 --target-kl 0.03 --learning-rate 1e-4 --std-min 0.3 --init-from ../checkpoints/best_stand_big1.zip >> ../checkpoints/handover_train_h4.log 2>&1
kill $NP
CK=$(ls -t ../checkpoints/autosave/ppo_handover_h4_*_steps.zip | head -1)
cp "$CK" ../checkpoints/handover_prone_h4.zip
env -u STANDAI_NO_RECOVERY_GATE ../.venv/bin/python -I analysis/controller_side_eval.py --handover none ../checkpoints/best_handover_prone.zip ../checkpoints/handover_prone_h4.zip --sides prone --levels 0.7 --n 200 --seed0 21000 --supine ../checkpoints/best_floor_supine_fdist.zip --prone ../checkpoints/best_prone8.zip --stand ../checkpoints/best_stand_big1.zip --workers 8 > ../checkpoints/analysis/ctl_prone_h4.out 2>&1
../.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from kakao_notify import send_message
txt = open("../checkpoints/analysis/ctl_prone_h4.out").read()
lines = [l[:120] for l in txt.splitlines() if "standing at" in l]
send_message("[h4 done] prone 20 s standing (seeds 21000+, 200)\n" + "\n".join(lines))
PY
