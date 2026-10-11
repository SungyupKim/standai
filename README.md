# standai

A 2D MuJoCo ragdoll biped learning to stand up and hold it, from an arbitrary starting posture,
via PPO (Stable-Baselines3). Split out of [`fightai`](../fightai) — the combat self-play league
this body was originally built for — once the standing problem turned out to need its own
rigorous track rather than being a side quest.

전체 배경과 상세 히스토리(한국어)는 [`docs/stand_only_track.md`](docs/stand_only_track.md),
축약된 영문 요약은 [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) 참고.

## Current status (2026-10-12)

The problem is now split into **two policies**, because the single stand policy never learned to get
off the floor (0/160 genuine uprights from starts below 0.6 m torso height, for every checkpoint):

| Policy | Job | State |
|---|---|---|
| **Stand policy** (`train_locomotion.py`, lineage `v21 -> v36 -> v37 -> v38`) | balance / hold a stand, recover from partial falls, **hold what the floor policy hands over** | `v37` regressed after ~100M steps (calm gate 51 -> 38-45 on 100 seeds at 102-112M), so it was stopped at 120M and `v38` restarted from `v37`@96M (calm 51) with 40% of episode starts drawn from 1,500 hand-off states. Running; first 2M steps: no change yet (hand-off 9/150 vs 11/150, calm 49 vs 51). |
| **Floor get-up policy** (`train_floor.py`, `FloorEnv`, fresh network, lineage `v1 -> v4`) | from a lying start, raise the body to a hand-off pose (torso >= 0.6 m, head >= 0.9 m, a foot down, pitch < 60 deg) | `v4` finished its 20M-step budget (25.9M cumulative): **reaches the hand-off pose in 80/100 deterministic fixed-seed starts** (median 1.8 s); 24% at 7.6M, 38% at 10.2M, 72-76% from 19M on. |

**Combined, end to end** (lying start -> floor policy -> stand policy for 3 s, 158 hand-off states from
200 starts): **8/100 starts end genuinely upright** with `v37`@99M (torso >= 0.7 m, head >= 0.9 m,
pitch < 40 deg held 20 steps), 3/100 with `v37`@120M. Before the split the same measurement was
0/160. The floor policy is no longer the bottleneck: the stand policy catches only ~7% of the
states handed over, because it never practised them -- which is what `v38` trains on. The switch
between the two policies and a learned hand-off boundary are **not built yet**; the numbers above come
from a hard-coded hand-off in test scripts (`scripts/make_handoff_pool.py`, `stand_progress_notify.py`).

Open problems, in order of importance: (1) does `v38` raise the hand-off catch rate (7% now; the
evaluation set of 150 hand-off states is disjoint from the 1,500 training states, so memorising shows
up as a train/eval gap) -- the pool is fixed, a refreshed or failure-weighted pool is the next step;
(2) the switch and hand-off boundary; (3) the floor policy reached the pose only from one of the two
start sides at 10M steps (38/82 vs 0/18, split by the sign of the initial torso pitch; which sign is
face-up is unverified) -- not re-checked at 25.9M; (4) PPO on the floor policy ran with `approx_kl` ~0.4-0.56
and `clip_fraction` ~0.78 the whole time, yet kept improving.

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
scripts/floor_env.py          FloorEnv: lying starts only, reward = head height; reports hand-off states
scripts/train_floor.py        PPO entrypoint for the floor get-up policy (fresh network)
scripts/floor_progress_notify.py  KakaoTalk pings with a policy-alone (deterministic) evaluation
scripts/watch_floor.py        Viewer for the floor policy (--only-success replays successes + 3 s after)
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
