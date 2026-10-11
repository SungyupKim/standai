# standai

A 2D MuJoCo ragdoll biped learning to stand up and hold it, from an arbitrary starting posture,
via PPO (Stable-Baselines3). Split out of [`fightai`](../fightai) — the combat self-play league
this body was originally built for — once the standing problem turned out to need its own
rigorous track rather than being a side quest.

전체 배경과 상세 히스토리(한국어)는 [`docs/stand_only_track.md`](docs/stand_only_track.md),
축약된 영문 요약은 [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) 참고.

## Current status (2026-10-11)

Fixed-seed, deterministic 3-second-survival gate (N=100 seeds; ±5 points of noise), current run
`v28` at ~95M steps, **on the x1.5 limb-torque body** (see below):

- **Calm starts** (near-standing posture): **~50%** (best so far; was 30–40% for all earlier runs).
  On 300 seeds: 152/300 vs 123/300 at the run's starting checkpoint on the same body.
- **Full distribution** (calm + harsh/collapsed starts mixed): **~25%**.
- **Harsh-only** (collapsed starts): **~6%**, unchanged. The policy essentially cannot get up from
  the ground yet -- that is the open problem.

What the investigation found (details, numbers and failures: [`docs/experiments_2026-10-11.md`](docs/experiments_2026-10-11.md)):

1. **The low harsh score is "cannot get up", not a measurement artifact.** Of collapsed starts that
   begin below the fall height, ~0-3% end up standing within the environment's 5 s window.
2. **The body can get up, but with little torque margin.** A closed-loop MPC oracle (exact simulator,
   diagnostic only) stands from 2/6 lying starts at the original actuator gears, and 5/6 at x1.5.
   An open-loop CEM search never did (it finds headstands and mid-air leaps). Hence the body spec
   changed to **x1.5 maximum limb torque** (`BODY_GEAR_SCALE` in `locomotion_env.py`; set
   `STANDAI_BODY_GEAR=1` to reproduce older numbers). Link masses were not scaled, so this is
   slightly optimistic for real hardware.
3. **Reward shaping and failed-start replay did not move harsh** (36M and 9M steps respectively); the
   body change moved calm.
4. Measurement lessons: training-log metrics disagree with the deterministic gate; torso-height-only
   objectives get exploited (inverted/airborne poses); 100-seed gates are too noisy for single
   comparisons.

Earlier history: for most of this track the real gate sat at a flat **0%** while training-log metrics
kept looking better; two silent measurement bugs (a stale module-level override, and a viewer that
never passed `stand_only=True`) and a restart from scratch (`v21`) ended that. The physical capability
of *holding* a stand was never in question (a `qfrc_bias` measurement shows it costs 1-5% of each
joint's maximum output).

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
docs/experiments_2026-10-11.md  Why harsh won't learn: measurements, negative results, MPC oracle, gear findings
scripts/analysis/             Diagnostics behind that doc (gates, MPC oracle, reward-by-height, gear sweep)
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
