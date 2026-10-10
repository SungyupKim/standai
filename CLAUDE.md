# standai

Working notes for whoever (human or Claude) picks this project up cold. Full history and
reasoning: [`docs/stand_only_track.md`](docs/stand_only_track.md) (Korean, detailed) and
[`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) (English, condensed). This file is just
the "how do I actually run/resume/check things" operational layer.

## What's running right now

Check before assuming anything is idle or stale:
```bash
pgrep -af "train_locomotion.py|stand_progress_notify.py|watch_locomotion.py"
```
Current lineage: `ppo_stand_v23_recoverygate`, logging to `checkpoints/locomotion_train_stand_v23.log`.
It's a `--init-from` continuation chain (v21 fresh-scratch -> v22 (abandoned, bad reward) ->
v23), not a single unbroken process -- the log file is append-only across restarts, so grep it
for the full history of this lineage's reward/curriculum changes.

## The one rule that matters most here

**Never trust the SB3 training-log metrics (`fall_rate`, `ep_len_mean`, etc.) by themselves.**
They're measured under stochastic (exploration-noise) actions over long episodes and can move in
the opposite direction of what you actually care about. Always re-measure directly:

```python
import sys; sys.path.insert(0, 'scripts')
import glob, os
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT

ckpt = max(glob.glob('checkpoints/autosave/ppo_stand_v23_recoverygate_*_steps.zip'), key=os.path.getmtime)
model = PPO.load(ckpt, device='cpu')

def gate(calm_frac, n=100):
    env = LocomotionEnv(stand_only=True)
    env.set_calm_frac(calm_frac)
    ok = 0
    for seed in range(n):
        obs, _ = env.reset(seed=seed)
        fell = False
        for t in range(150):  # 150 steps = 3s at 0.02s/step
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT:
                fell = True; break
            if term or trunc: break
        if not fell: ok += 1
    return ok

print('calm gate:', gate(1.0), '/100')       # near-standing starts only
print('full-dist gate:', gate(None), '/100') # calm+harsh mix (calm_frac=None keeps env default)
print('harsh-only gate:', gate(0.0), '/100') # force every reset to a collapse start
```
(`set_calm_frac(None)` isn't real -- call `gate()` with the third variant by just not calling
`set_calm_frac` at all if you need the env's own default mix.)

## Resuming training after a restart/reboot

```bash
cd scripts
LATEST=$(ls -t ../checkpoints/autosave/ppo_stand_v23_recoverygate_*_steps.zip | head -1)
nohup ../.venv/bin/python train_locomotion.py --n-envs 8 --timesteps 60000000 --stand-only \
  --out ppo_stand_v23_recoverygate \
  --height-start-scale 6.0 --height-end-scale 15.0 --height-relax-frac 0.3 --height-already-relaxed \
  --curriculum-fall-gate 0.65 --curriculum-streak-gate 350 \
  --init-from "$LATEST" \
  >> ../checkpoints/locomotion_train_stand_v23.log 2>&1 &
disown

nohup ../.venv/bin/python stand_progress_notify.py --log ../checkpoints/locomotion_train_stand_v23.log \
  --ckpt-glob '../checkpoints/autosave/ppo_stand_v23_recoverygate_*_steps.zip' \
  --total-timesteps 60000000 --interval-minutes 30 >> ../checkpoints/stand_progress_notify_v23.log 2>&1 &
disown
```
`--height-already-relaxed` matters: the height-reward scaffold has already ramped to 15.0; without
this flag a continuation re-ramps it from 6.0. `--curriculum-fall-gate`/`--curriculum-streak-gate`
are the loosened phase-1->phase-2 gate (default 0.1/500 is close to unreachable in practice --
0.65/350 matches what this lineage's calm performance actually achieves).

**Reward-constant-only changes continue via `--init-from` on the latest checkpoint -- don't
restart from scratch.** The one exception found so far: five-plus sequential `--init-from` warm
starts in a row seemed to destabilize the value function (v19->v20 lineage); if behavior looks
qualitatively wrong after several chained changes, consider rolling back to a known-good
checkpoint rather than continuing to chain.

## Watching a checkpoint

```bash
cd scripts
ckpt=$(ls -t ../checkpoints/autosave/ppo_stand_v23_recoverygate_*_steps.zip | head -1)
DISPLAY=:1 ../.venv/bin/python watch_locomotion.py "$ckpt"
```
**`DISPLAY=:1`, not `:0`** -- this machine's X server runs on `:1` (`who` shows it). Resets are
random every episode (no fixed seed, random calm/harsh mix) -- this is for a qualitative look,
not a measurement; use the gate snippet above for anything you intend to report as a number.

If the viewer fails with a GLX/OpenGL error: `nvidia-smi` -- if it says
`Failed to initialize NVML: Driver/library version mismatch`, a background system update bumped
the driver without a reboot. Training itself (CUDA path) is unaffected; only the viewer's GL
context creation breaks. Fix needs a reboot -- note that Claude Code itself runs on this same
machine, so a reboot kills the current session too. Before rebooting: autosaves happen every
25,000 steps, so loss on restart is trivial; just re-run the resume command above afterward.

## Files

See [`README.md`](README.md) for the project layout. `build_model.py`/`env.py`/`view.py`/
`kakao_notify.py` are copies of `../fightai`'s versions, not a shared dependency -- a body change
in one repo needs the same edit applied by hand in the other.
