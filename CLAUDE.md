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
Current lineage: `ppo_stand_v28_body15`, logging to `checkpoints/locomotion_train_stand_v28.log`.
**Body spec changed (2026-10-11): limb actuator gears are x1.5 by default** (`BODY_GEAR_SCALE` in
`locomotion_env.py`, hip 270 / knee 90 / ankle 60; root_x/root_ry untouched). Reason: a closed-loop MPC
oracle stands from only 2/6 lying starts at x1 but 5/6 at x1.5 (same as x2), and an unchanged policy
gains on crouched calm starts at x1.5 without losing upright ones. **Every result recorded before v28
(including all "baselines" below: calm 28-40 / full 17-23 / harsh 5-7) is at x1.** To reproduce an old
number run the script with `STANDAI_BODY_GEAR=1`. The training/gate/notifier/viewer all pick the body up
from the env automatically; never compare gate numbers across body specs.
v28 branched from the v26 step-91.45M checkpoint (calm 37/full 18/harsh 6 at x1; 41/22/5 at x1.5), pure PPO,
fixed x1.5 body, calm_frac=0.6 (`--calm-end-frac 0.6`, no curriculum flags). It replaces v27 (gear
curriculum [1,2]->1, stopped at 93.3M after ~1.8M steps: gear_max 2.0->1.95). Abandoned: v22/v23
(plateau at x1), v25 (`--getup-reward`, 36M steps, no change), v26 (`--replay-curriculum`, 9M steps, no
change). `--getup-reward` / `--ground-shaping` / `--replay-curriculum` / `--gear-curriculum` remain in the
code (opt-in). Full write-up: `docs/experiments_2026-10-11.md`.

Why this approach (findings, 2026-10-11):
- Harsh-only is ~entirely "cannot get up from the ground": of harsh starts that begin below
  FALL_HEIGHT only ~0% end standing even with the env's full 5s (DOWN_RECOVERY_STEPS=250).
- Calm is NOT uniformly solved: starts with knee >= -10 deg hold 82%, -25..-10 54%, -40..-25 28%,
  worse beyond -60 deg ~0%. The hard end of "calm" is the first rung of the ladder.
- The body CAN get up: closed-loop MPC (predictive sampling, exact physics) stood up from flat
  starts in ~1.5s (2/6 starts at gear x1); open-loop CEM never did (headstands / mid-air leaps /
  sitting). MPC is a diagnostic here, not a teacher.
- Torque IS a major bottleneck: the same MPC stands 5/6 at gear x2 (first upright in 0.4-0.8s), including
  the flattest starts that fail at x1; the unchanged v26 policy gains ~+20 points on crouched calm starts
  at gear x1.5 (but degrades at x2+ since it was trained at x1). Hence the x1.5 body spec (v28).
  Per-joint MPC sweep (only one group x1.5): knee 2/6, hip 1/6, hip+knee 2-3/6, ankle 1/6 vs 5/6 for all
  limbs -- no single group reproduces it (waist/arms untested; n=6 is noisy). v28 early result (94.9M):
  calm gate 52 (prev best 41), 300-seed calm 152 vs 123 at its start; harsh unchanged (5/197 end standing).
- recovery_gate (knee_straight/torso_upright off when low) is fine: v22 (no gate) vs v23 (gate) from
  the same init gave the same gates. Do not revert it.
- Literature (Tao+ 2022 "Learning to Get Up", HumanUP 2025): curriculum + staged difficulty; HumanUP
  used ~5B steps, i.e. ~100x what we run. Strong-body-first is what v27 tries;
  a canonical-pose ladder (squat/kneel/sit/lie) is not yet tried.
- Judge v28 by the gate snippet below (x1.5 body, the env default). Start-of-run baseline for v28 is its
  init checkpoint on the same body: calm 41 / full 22 / harsh 5 (100 seeds). Also the 300-seed knee-bin
  table of calm starts (crouched starts are the sensitive part).

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

ckpt = max(glob.glob('checkpoints/autosave/ppo_stand_v28_body15_*_steps.zip'), key=os.path.getmtime)
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
LATEST=$(ls -t ../checkpoints/autosave/ppo_stand_v28_body15_*_steps.zip | head -1)
nohup ../.venv/bin/python train_locomotion.py --n-envs 8 --timesteps 60000000 --stand-only \
  --out ppo_stand_v28_body15 \
  --height-start-scale 6.0 --height-end-scale 15.0 --height-relax-frac 0.3 --height-already-relaxed \
  --curriculum-fall-gate 0.65 --curriculum-streak-gate 350 \
  --calm-already-relaxed --calm-end-frac 0.6 \
  --init-from "$LATEST" \
  >> ../checkpoints/locomotion_train_stand_v28.log 2>&1 &
disown

nohup ../.venv/bin/python stand_progress_notify.py --log ../checkpoints/locomotion_train_stand_v28.log \
  --ckpt-glob '../checkpoints/autosave/ppo_stand_v28_body15_*_steps.zip' \
  --total-timesteps 60000000 --interval-minutes 30 >> ../checkpoints/stand_progress_notify_v28.log 2>&1 &
disown
```
`--calm-already-relaxed` matters just as much: without it a restart resets `calm_frac` to 1.0 (all
calm) and re-anneals down, silently cutting harsh exposure for ~9M steps. `--calm-end-frac` sets the
harsh/calm target (0.6 for v28; the env default is `CALM_EPISODE_FRAC` = 0.45). `--gear-curriculum` must be
passed on every v27 resume (and `--gear-start` set to the last logged `gear/max`).
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
ckpt=$(ls -t ../checkpoints/autosave/ppo_stand_v28_body15_*_steps.zip | head -1)
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
