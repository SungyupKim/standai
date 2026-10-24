# standai

A 2D MuJoCo ragdoll biped learning to stand up and hold it, from an arbitrary starting posture,
via PPO (Stable-Baselines3). Split out of [`fightai`](../fightai) — the combat self-play league
this body was originally built for — once the standing problem turned out to need its own
rigorous track rather than being a side quest.

전체 배경과 상세 히스토리(한국어)는 [`docs/stand_only_track.md`](docs/stand_only_track.md),
축약된 영문 요약은 [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) 참고.

> ## Result: a ragdoll lying on the floor stands up and is still standing 20 s later in **about 68%** of starts
>
> A rule-based controller chains four PPO-trained policies (supine floor policy, prone floor policy, a hand-over policy on the prone side, stand policy). The supine floor policy and the stand policy are 256x256 nets distilled from the earlier 64x64 policies and fine-tuned; the supine floor policy was additionally trained with a continuous *arrival-quality* reward (`checkpoints/best_floor_supine_fdist.zip`, `best_prone8.zip`, `best_stand_big1.zip`, `best_handover_prone.zip`, hand-over region level 0.7).
> **~68%** standing after 20 s on fresh seed sets: supine start ~53-58% (925/1600 = 57.8% over eight sets of 200 starts, 46-62% per set; 53% on one independent pair of sets), prone start ~83% (1327/1600 over eight sets, 79-87% per set); the 50:50 average is ~70% pooled and 66.6% on the one independent pair of sets. At the start of the project this was **0/160**.
> Caveats: simulation only; the 50:50 supine/prone weighting is an assumption (the environment's own 80:20 split is an artefact of its collapse generator, docs 3.20); +-2.5 points from sampling, about +-4-5 points between seed sets and between runs of the same seeds, and the seed sets that were used to choose between candidate policies are slightly optimistic, and 1.2% of prone starts are not lying down (the start resampling gives up after 30 tries; ~+0.2 points on the prone side, not corrected). The supine side is the bottleneck (16% of starts never reach the hand-over region and about a third of those that do fall back within a second, docs 3.24 addendum 10). Reproduce: `.venv/bin/python -I scripts/analysis/controller_side_eval.py --supine checkpoints/best_floor_supine_fdist.zip --prone checkpoints/best_prone8.zip --stand checkpoints/best_stand_big1.zip --handover checkpoints/best_handover_prone.zip --sides supine prone --levels 0.7 --n 200 --seed0 63000` (a shell that exports training variables such as `STANDAI_NO_RECOVERY_GATE` or `STANDAI_HARSH_MIN_TORSO` contaminates the result: use `env -u` / `env VAR=...`).

![Demo: selected successful runs of the whole system in simulation (supine and prone starts)](assets/demo.gif)

*Four selected successful runs (two supine, two prone starts) of the whole system: floor policy -> hand-over policy (prone side) -> stand policy (the current 256x256 system: `best_floor_supine_fdist`, `best_prone8`, `best_stand_big1`, `best_handover_prone`). They are picked from seeds that end standing without ever falling on the way (some other starts stand, fall and get up again, or never get up); the overall rate is ~68% (supine ~53-58%, prone ~80-86%, fresh seed sets of 200-1600 starts per side), so most other starts end differently. Higher quality: [`assets/demo.mp4`](assets/demo.mp4). Regenerate (headless, no display needed): `python scripts/render_demo.py`.*

## Current status (2026-10-25)

The problem is split into **four policies and a rule-based controller** (`scripts/controller.py`, `GetUpController`); the single stand policy never learned to get off the floor (0/160 genuine uprights from starts below 0.6 m). The sign of the initial torso pitch (observable) picks the supine or prone floor policy; the floor policy hands over once the body is in the *hand-over region* (foot down, head >= 0.9 m, torso >= 0.6 m, slow; strictness level 0.7); on the prone side a hand-over policy sits in between; the stand policy then holds the stand, and if it falls (torso < 0.45 m) control goes back to the floor policy. **`scripts/analysis/benchmark.py` and `controller_side_eval.py` measure the whole system per start side.**

| Policy | Job | State |
|---|---|---|
| **Supine floor** (`train_floor.py`, `FloorEnv`, `best_floor_supine_fdist.zip`, 256x256) | lying on the back -> hand-over region | distilled from the 64x64 `v12`, +40M and +100M PPO steps, then +40M with the arrival-quality reward; reaches the region from ~84% of starts |
| **Prone floor** (`best_prone8.zip`, 64x64) | lying face down -> hand-over region | reaches the region from ~90% of starts |
| **Hand-over policy, prone side** (`train_handover.py`, `best_handover_prone.zip`) | from the prone arrival to a stand-ready, balanced state | warm-started from the stand policy, trained on arrivals ordered by how far the centre of mass is ahead of the front foot; prone 20 s standing 70% -> 83% |
| **Stand policy** (`train_locomotion.py`, `best_stand_big1.zip`, 256x256) | hold the stand from what the others hand over | distilled from the 64x64 `v55b` and fine-tuned at low learning rate (5e-5, `--target-kl`) |

What moved the end-to-end number (controller, 20 s standing, 50:50 supine/prone; every step is documented with its failures in [`docs/experiments_2026-10-11.md`](docs/experiments_2026-10-11.md) 3.17-3.24 and [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md)):

| Date | Change | 50:50 |
|---|---|---|
| 10-17 | rule-based controller, stand policy `v37` | ~24% |
| 10-18 | stand run at a low learning rate (the earlier runs were jumping by +-50 catches between checkpoints, not improving); hand-over region level 0.7 | ~50% |
| 10-19 | separate hand-over policy on the prone side, stand policy `v55b` (COM-ahead ladder) | ~62.5% |
| 10-22 | stand and supine floor policy distilled into 256x256 nets and fine-tuned (big nets from scratch always failed) | ~65% |
| 10-23 | hand-over policy re-trained from the big stand policy | prone 83% |
| 10-24 | supine floor policy paid for the *quality of its arrival* (COM forward speed, low pitch) | supine ~49% -> ~58%, ~68% |

What did not move it (all measured on fresh seeds, several of them looked promising in a training log): more floor training that raises the hand-over region reach rate by 13 points (+3 points of standing), re-training the stand policy on the new arrivals, three supine hand-over policies, delaying the hand-over until the centre of mass moves forward, stand-policy critic values as an auxiliary floor reward, stronger arrival rewards, and a reward from the stand policy's real catch outcome at the arrival. Reach rate is not a selection criterion for 20 s standing.

Open problems, in order of importance: (1) the supine side: 16% of starts never reach the region, and about a third of the arrivals fall back within a second (the arrivals that stand carry a forward COM speed of ~+0.6 m/s, the others ~+0.3 m/s or 0, and fall backwards); (2) the rule-based switching could be replaced by one network (distil the controller, then fine-tune end to end; not tried); (3) crouched starts remain the stand policy's weak spot.

A measurement trap that recurs: the trainer's `fall_rate` and `ep_len_mean` are optimistic or unrelated to the gate; judge with fixed seeds per start side. The same seeds move by +-4 points between runs and 46-62% between seed sets on the supine side, so small differences need 600+ starts and a paired comparison.

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
scripts/make_handoff_pool.py / make_strict_pool.py  collect hand-off states (--supine-only, --level) for training the stand / hand-over policies
scripts/controller.py         GetUpController: rule-based combination of the supine / prone floor policies, the prone-side hand-over policy and the stand policy
                              (opt-in diagnostics: STANDAI_VX_GATE hand-over delay; scripts/analysis/controller_side_eval.py, funnel.py, benchmark.py)
scripts/handover_env.py / train_handover.py  hand-over policy environment and trainer (warm start from the stand policy; arrival ladder)
scripts/distill_stand.py / distill_floor.py  behaviour cloning + value regression + DAgger: 64x64 policy -> 256x256, then PPO fine-tune
scripts/floor_trend_notify.py / heartbeat.py KakaoTalk progress for floor runs (reach rate, arrival speed/pitch, catch rate) / 30-minute pipeline heartbeat
scripts/render_demo.py        headless demo video of the current system (assets/demo.mp4/gif; picks clean successful runs)
scripts/overnight_*.sh        unattended pipelines (distill -> fine-tune -> scan -> controller comparison -> Kakao message)
scripts/analysis/             benchmark / controller_side_eval / funnel / supine_fall_diag (state at the hand-over by outcome) / catch_by_com / floor_ckpt_scan ...
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

**The whole system (viewer and benchmark):**
```bash
DISPLAY=:1 .venv/bin/python scripts/watch_controller.py --side alternate --level 0.7 \
  --supine checkpoints/best_floor_supine_fdist.zip --prone checkpoints/best_prone8.zip \
  --stand checkpoints/best_stand_big1.zip --handover checkpoints/best_handover_prone.zip
env -u STANDAI_NO_RECOVERY_GATE .venv/bin/python -I scripts/analysis/controller_side_eval.py --n 200 --seed0 63000 \
  --supine checkpoints/best_floor_supine_fdist.zip --prone checkpoints/best_prone8.zip --stand checkpoints/best_stand_big1.zip \
  --handover checkpoints/best_handover_prone.zip --sides supine prone --levels 0.7
```

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

## License

MIT, see [`LICENSE`](LICENSE). `build_model.py`, `env.py`, `view.py` and `kakao_notify.py` are copies of files from the author's [`fightai`](https://github.com/SungyupKim/fightai) project.
