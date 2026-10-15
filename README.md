# standai

A 2D MuJoCo ragdoll biped learning to stand up and hold it, from an arbitrary starting posture,
via PPO (Stable-Baselines3). Split out of [`fightai`](../fightai) — the combat self-play league
this body was originally built for — once the standing problem turned out to need its own
rigorous track rather than being a side quest.

전체 배경과 상세 히스토리(한국어)는 [`docs/stand_only_track.md`](docs/stand_only_track.md),
축약된 영문 요약은 [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) 참고.

## Current status (2026-10-15)

The problem is split into **three policies**, because the single stand policy never learned to get off the floor
(0/160 genuine uprights from starts below 0.6 m torso height, for every checkpoint). The sign of the initial torso pitch
(observable) picks the floor policy; both hand over to the stand policy once the body is in the *stand-ready region*
(foot down, torso >= 1.08 m, knees > -20 deg, |pitch| < 20 deg, |vz| < 1 m/s, |pitch rate| < 100 deg/s).

| Policy | Job | State (deterministic, fixed seeds) |
|---|---|---|
| **Supine floor policy** (`train_floor.py`, `FloorEnv`, lineage `v1 -> v4 -> ... -> v12`) | lying on the back (~80% of lying starts) -> stand-ready region | reaches the region from **55%** of its starts; the stand policy then holds 51% of those for 3 s (**~28%** of starts) |
| **Prone floor policy** (`--prone-frac 1.0`, lineage `prone3 -> prone4 -> prone8`) | lying face-down (~20%; inferred from the pitch sign, unverified) -> region | 0% with a plain restart; **~58%** reach with a staged target (loose -> strict, `--level-start 0.0`); stand policy holds 36% of those (**~22%** of starts); plateau since ~77M steps |
| **Stand policy** (`train_locomotion.py`, `v37`@96M) | hold the stand from what the floor policies hand over, balance, recover from partial falls | holds 77-92% of upright starts but only 36-51% of hand-offs and 5-44% of crouched starts. A from-scratch retrain on a broader start mix (upright + crouched + pooled hand-off states) is being prepared |

**End to end, floor policy -> stand policy (natural start mix, 20 s): ~26% end up standing.** Loop back to the floor
policy when the stand policy falls: 24% (no gain). Floor policies alone: 4%. The switch / selector between the
three models is **not built**; the numbers above come from hard-coded hand-offs in `scripts/analysis/system_loop.py`
and `hold_compare.py`. Before the split the same measurement was 0/160.

Open problems, in order of importance: (1) the stand policy catches too few hand-off states (the centre of mass being
~10 cm ahead of the front foot is what separates the ones it loses; training the floor policy to avoid it did not help,
so the stand policy has to learn them); (2) the crouched-start gap (nobody owns it); (3) the switch / selector;
(4) every result is a single training run; (5) PPO on the floor policies runs with `approx_kl` 0.1-0.7 and checkpoints
after the peak regress, so checkpoints are picked by re-measuring a few candidates on 400 seeds.

What the investigation found (details, numbers and failures: [`docs/experiments_2026-10-11.md`](docs/experiments_2026-10-11.md)):

1. **The low harsh score is "cannot get up", not a measurement artifact.** Of collapsed starts that
   begin below the fall height, ~0-3% end up standing within the environment's 5 s window.
2. **The body can get up, but with little torque margin.** A closed-loop MPC oracle (exact simulator,
   diagnostic only) stands from 2/6 lying starts at the original actuator gears, and 5/6 at x1.5.
   An open-loop CEM search never did (it finds headstands and mid-air leaps). Hence the body spec
   changed to **x1.5 maximum limb torque** (`BODY_GEAR_SCALE` in `locomotion_env.py`; set
   `STANDAI_BODY_GEAR=1` to reproduce older numbers). Link masses were not scaled, so this is
   slightly optimistic for real hardware. The floor policy confirms the torque point: 65% of its
   deterministic actions sit at the +-1 limit.
3. **Reward shaping, failed-start replay, correlated noise, bursts and a frontier-ladder reset archive
   did not make the single policy get up from the floor** (policy-alone head > 0.5 m stayed at 4-8% of
   floor starts, below random bang-bang patterns at 77%). A fresh network trained on floor starts only
   did: 72% at 0.6M steps without bursts, whereas env-level bursts *hurt* it (20% at 1.3M steps).
   Getting to the *stand-ready* region (not just head height) needed the right target: staged targets
   (loose -> strict) worked on top of a network that already knew how to get high, and failed from scratch.
4. Measurement lessons: training-log metrics disagree with the deterministic gate; torso-height-only
   objectives get exploited (inverted/airborne poses); 100-seed gates are too noisy for single
   comparisons; **a "success" has to be checked by what happens next** -- the floor policy's first
   success definition (head >= 0.9 m once) turned out to be a pop-up the same policy could not hold,
   and the check that "proved" it continued the *wrong* policy (the stand policy takes over after
   hand-off). Both errors are written up in the experiments doc.

Earlier history: for most of this track the real gate sat at a flat **0%** while training-log metrics
kept looking better; two silent measurement bugs (a stale module-level override, and a viewer that
never passed `stand_only=True`) and a restart from scratch (`v21`) ended that. The physical capability
of *holding* a stand was never in question (a `qfrc_bias` measurement shows it costs 1-5% of each
joint's maximum output).

Limits to keep in mind: 2D sagittal-plane model, limb torque raised to x1.5 without scaling link
masses, and ~100M steps of training (get-up papers use ~5B). Much of the code and analysis was written
in a pair-programming loop with Claude (an AI assistant); experiment design decisions and review were
the repository owner's.

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
scripts/floor_env.py          FloorEnv: lying starts only; targets: loose hand-off / stand-ready region (standready), staged level 0..1,
                              prone-only starts (prone_frac), optional COM condition (com_max), 5 s KO rule off (down_timeout)
scripts/train_floor.py        PPO entrypoint for the floor get-up policies (--prone-frac, --level-start, --no-down-timeout, --max-steps, ...)
scripts/floor_progress_notify.py  KakaoTalk pings: policy-alone eval, split by start side, stand policy taking over for 3 s
scripts/watch_floor.py        Viewer for the floor policy (--stand <ckpt> hands over, --only-success --skip-replay shows only the catch,
                              --lying-only --prone-frac 1.0 prone starts only)
scripts/make_handoff_pool.py / make_strict_pool.py  collect hand-off states for training the stand policy (loose / strict region)
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

**Training the prone floor policy (staged target, warm start, strict region, lying face-down only):**
```bash
cd scripts
../.venv/bin/python train_floor.py --n-envs 8 --timesteps 150000000 --out ppo_floor_prone8 --floor-burst-rate 0.0 \
  --standready --crouch-frac 0.0 --max-steps 450 --prone-frac 1.0 --no-down-timeout --speed-pen 0.0 --air-pen 0.0 \
  --learning-rate 5e-5 --init-from ../checkpoints/best_prone8.zip
# staged target: add --level-start 0.0 --level-gate 0.35 --level-step 0.05 (needs a network that can already get high; see docs)
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
