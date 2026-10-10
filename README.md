# standai

A 2D MuJoCo ragdoll biped learning to stand up and hold it, from an arbitrary starting posture,
via PPO (Stable-Baselines3). Split out of [`fightai`](../fightai) — the combat self-play league
this body was originally built for — once the standing problem turned out to need its own
rigorous track rather than being a side quest.

전체 배경과 상세 히스토리(한국어)는 [`docs/stand_only_track.md`](docs/stand_only_track.md),
축약된 영문 요약은 [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) 참고.

## Current status (2026-10-10)

Fixed-seed, deterministic 3-second-survival gate, measured fresh at 55.3M steps into the current
run (`v21`, started from scratch two days ago):

- **Calm starts** (near-standing posture, N=100, deterministic): **~30–40%** survive 3 full
  seconds — noisy run to run, trending up.
- **Full distribution** (calm + harsh/collapsed starts mixed, N=100): **~17–23%**.
- Training-log `fall_rate` has dropped from ~1.0 to ~0.55–0.75 over the run; `max_stand_streak`
  (longest continuous stand within an episode) is up to 300–480 steps (6–10s), from ~20 at the
  start.

For most of this track's history the real gate sat at a flat **0%** — training-log metrics kept
looking better while the thing actually being measured (stand up from a fixed seed, hold it for
3 seconds, deterministic policy) never budged. Two things changed that:

1. **Found two silent measurement bugs**: a stale module-level override that was quietly
   overwriting a reward-scale tuning attempt, and a viewer script that had never actually been
   passing `stand_only=True` — every manual "does this look right" check all night had secretly
   been running with an assist force re-enabled that training doesn't get.
2. **Abandoned a long chain of `--init-from` warm-started reward-constant changes and restarted
   from scratch** with everything (body proportions, the full curriculum, current reward scales)
   baked in from step 0. The suspicion was that five-plus sequential warm starts had destabilized
   the value function. The scratch run (`v21`) hit its first non-zero real-gate result (5%) in
   under 12M steps — the previous warm-started lineage took 33M+ steps to reach 1%. That result
   held up as training continued, so the "too much accumulation" hypothesis looks correct, at
   least in this instance.

Curriculum is still holding at `calm_frac=1.0` (100% near-standing starts) — the performance gate
to start mixing in harsh/collapsed starts (`fall_rate ≤ 0.1` and `max_stand_streak ≥ 500` steps,
sustained over 200 episodes) hasn't been cleared yet. The physical capability was never in
question: a direct `qfrc_bias` torque measurement at a hand-set standing pose shows it costs only
1–5% of each joint's maximum output to hold a full standing posture. What's missing is still
reward/curriculum design, not body capability.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install mujoco gymnasium stable-baselines3 tensorboard
```

`watch_locomotion.py` needs a machine with an OpenGL context. Training itself runs headless.

## Project layout

```
models/fighter2d.xml          Generated MJCF model (build_model.py builds it — copied from fightai)
scripts/build_model.py        Model generator (copied from fightai, shared body definition)
scripts/env.py                Fighter2DEnv base class (copied from fightai; LocomotionEnv subclasses it)
scripts/locomotion_env.py     LocomotionEnv: the stand/recover environment (stand_only mode)
scripts/train_locomotion.py   PPO training entrypoint, performance-gated curriculum
scripts/watch_locomotion.py   Interactive MuJoCo viewer for a trained checkpoint
scripts/stand_progress_notify.py  Periodic KakaoTalk progress pings with a real fixed-seed gate
scripts/kakao_notify.py       Shared KakaoTalk notify helper (copied from fightai)
checkpoints/                  Trained models, autosaves, logs (gitignored)
docs/                         Technical documentation (Korean full history + English summary)
```

`build_model.py`/`env.py` are copies of fightai's versions, not a shared dependency — if the body
definition changes, both repos need the change applied by hand. See fightai's README for why this
split was done by copy rather than a shared submodule.

## Running things

**Training (learn to stand from any posture):**
```bash
cd scripts
../.venv/bin/python train_locomotion.py --n-envs 8 --timesteps 80000000 --stand-only --out my_stand
```

**Resuming from a checkpoint:**
```bash
../.venv/bin/python train_locomotion.py --n-envs 8 --timesteps 80000000 --stand-only \
  --out my_stand --init-from ../checkpoints/autosave/my_stand_..._steps.zip
```

**Watching a trained policy:**
```bash
DISPLAY=:1 ../.venv/bin/python watch_locomotion.py ../checkpoints/autosave/my_stand_..._steps.zip
```

**Progress notifications (KakaoTalk, every 30 min, includes a real fixed-seed gate measurement):**
```bash
nohup ../.venv/bin/python stand_progress_notify.py \
  --log ../checkpoints/locomotion_train_my_stand.log \
  --ckpt-glob '../checkpoints/autosave/my_stand_*_steps.zip' \
  --total-timesteps 80000000 --interval-minutes 30 &
```

## Methodology note

The single most important lesson carried over from fightai: SB3's own training-log metrics
(`fall_rate`, `ep_len_mean`, etc.) are measured under stochastic rollout over long episodes and can
look good or bad independently of what actually matters — whether the policy, acting
deterministically from a fresh reset, stays upright for a full 3 seconds. Every progress check in
this repo re-measures that directly (`stand_progress_notify.py`'s `gate_survival_rate*` functions)
rather than trusting the log.

## Roadmap

Standing (from any posture, held reliably) → walking (reintroducing velocity tracking) → transplant
the resulting policy back into fightai's combat action/observation space via
`transplant_locomotion.py` (still in fightai) — each stage gated on the previous one actually
working.
