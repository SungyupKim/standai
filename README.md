# standai

A 2D MuJoCo ragdoll biped that learns, with PPO (Stable-Baselines3), to get up from lying on its back or face down and stay standing. Split out of [`fightai`](../fightai), the combat self-play league this body was built for, once standing turned out to need its own rigorous track.

Technical details: [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) (English, condensed), [`docs/experiments_2026-10-11.md`](docs/experiments_2026-10-11.md) (Korean, every experiment with its failures), [`docs/stand_only_track.md`](docs/stand_only_track.md) (Korean, full history). Working notes for resuming the project: [`CLAUDE.md`](CLAUDE.md). **Trained checkpoints (the single network, size ablation, the four rule-based components): [huggingface.co/SungyupKim/standai-getup](https://huggingface.co/SungyupKim/standai-getup).**

> ## Result: one network gets up from lying on the back or face down and is still standing 20 s later in **about 94%** of starts (from 0/160 at the start of the project)
>
> Fresh seed sets, 600 starts per side: **supine start 97.7%, prone start 90.7%, 50:50 average 94.2%** (5 s of uninterrupted standing at the end: 97.0% / 90.0%); the rule-based controller of four PPO policies that it was distilled from stands in 70.8% on the same seeds (supine 63.0%, prone 78.7%). Training recipe: build the pieces with their own rewards and curricula, distil the whole controller (switching rules included) into ONE 512x512 network with DAgger, then fine-tune it end to end with a 20 s "standing" reward (docs 3.24 addendum 13).
> Checks against shortcuts (all on fresh seeds): no use of the unphysical `root_x`/`root_ry` thrust (applied control is 0), standing unchanged when the parked opponent body is moved 40 m away, 5 s of calm standing (torso height std 0.009 m), and the stand is not knocked over by 150-500 N pushes (supine 94% -> 93%, prone 88% -> 88%).
> Caveats: simulation only, 2D; starts come from the environment's own lying-start generator, so generalisation to other start poses is untested; the 50:50 supine/prone mix is an assumption; +-2.5 points sampling error; 1.2% of prone starts are not lying down (start resampling gives up after 30 tries, ~+0.2 points, not corrected). Until 2026-10-25 the evaluation also applied the prone hand-over policy to supine arrivals, so older figures of ~65-68% for the rule-based system were ~3 points too low.

![Demo: selected successful runs of the whole system in simulation](assets/demo.gif)

*Four selected runs of the single network (two supine, two prone starts; no fall or sit-back on the way). They are picked, not typical: about 98% of supine and 91% of prone starts end standing. Higher quality: [`assets/demo.mp4`](assets/demo.mp4); regenerate headless with `python scripts/render_demo.py --student checkpoints/best_getup_e2e1.zip --seed0 98100`.*

## How it works

The sign of the initial torso pitch picks the supine or prone **floor policy**; when the body reaches the *hand-over region* (foot down, head >= 0.9 m, torso >= 0.6 m, slow; strictness level 0.7) control passes to the **stand policy** (on the prone side via a **hand-over policy**); if the stand policy falls (torso < 0.45 m) the floor policy takes over again. One network could not learn to get off the floor (0/160 genuine uprights from starts below 0.6 m); a fresh network trained on floor starts only did.

| Policy | Job | Checkpoint |
|---|---|---|
| Supine floor (`train_floor.py`, 256x256) | lying on the back -> hand-over region; trained with a continuous *arrival-quality* reward | `best_floor_supine_fdist.zip` |
| Prone floor (64x64) | lying face down -> hand-over region | `best_prone8.zip` |
| Hand-over, prone side (`train_handover.py`) | prone arrival -> stand-ready, balanced state | `best_handover_prone.zip` |
| Stand (`train_locomotion.py`, 256x256) | hold the stand from what the others hand over | `best_stand_big1.zip` |

What moved the 50:50 number (every step with its failures is in the docs):

| Date | Change | 50:50 |
|---|---|---|
| 10-17 | rule-based controller | ~24% |
| 10-18 | low PPO learning rate for the stand policy; hand-over region level 0.7 | ~50% |
| 10-19 | hand-over policy on the prone side | ~62% |
| 10-22 | stand and supine floor policy distilled into 256x256 nets, fine-tuned | ~65% (as measured then) |
| 10-24 | supine floor policy rewarded for the quality of its arrival (COM forward speed, low pitch) | +4 points |
| 10-25 | evaluation corrected (rule-based system ~71-73%); whole controller distilled into one network, then PPO end to end | rules ~72% -> **one network ~94%** |

What did not move it (fresh seeds; several looked promising in a training log): more floor training that raises the reach rate 13 points (+3 points of standing), re-training the stand policy on the new arrivals, supine hand-over policies, delaying the hand-over by a learned or hand-made score, a critic value or the stand policy's real catch outcome as floor reward, stronger arrival rewards. Open problems: generalisation of the single network to other start poses and bodies, whether it also works from scratch at a large size (expected not: big networks trained from scratch failed on the floor policies). Size ablation (same recipe, 12M end-to-end steps, 50:50): 512x512 92%, 256x256 93%, 128x128 89%, 64x64 85% vs 72% for the rules, i.e. the recipe matters more than the size.

Limits: 2D sagittal-plane model, limb torque x1.5 without scaling link masses (`BODY_GEAR_SCALE`), ~1B steps in total (get-up papers use ~5B). Much of the code and analysis was written in a pair-programming loop with Claude (an AI assistant); experiment design decisions and review were the repository owner's.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install mujoco gymnasium stable-baselines3 tensorboard
```
Viewers need an OpenGL context (`DISPLAY=:1` on this machine); training and `render_demo.py` run headless.

## Running things

```bash
# watch the whole system, then measure it (paired, fixed seeds, per start side)
DISPLAY=:1 .venv/bin/python scripts/watch_controller.py --side alternate --level 0.7 \
  --supine checkpoints/best_floor_supine_fdist.zip --prone checkpoints/best_prone8.zip \
  --stand checkpoints/best_stand_big1.zip --handover checkpoints/best_handover_prone.zip
env -u STANDAI_NO_RECOVERY_GATE .venv/bin/python -I scripts/analysis/controller_side_eval.py --n 200 --seed0 63000 \
  --supine checkpoints/best_floor_supine_fdist.zip --prone checkpoints/best_prone8.zip --stand checkpoints/best_stand_big1.zip \
  --handover checkpoints/best_handover_prone.zip --sides supine prone --levels 0.7

# the single network and its end-to-end fine-tune
.venv/bin/python scripts/distill_controller.py            # DAgger from the rule-based controller
.venv/bin/python scripts/train_getup.py --init-from checkpoints/getup_student_distilled.zip --out e2e1
.venv/bin/python -I scripts/analysis/getup_eval.py --arms teacher checkpoints/getup_student_distilled.zip --sets 91000 92000
```
A shell that exports training variables (`STANDAI_NO_RECOVERY_GATE`, `STANDAI_HARSH_MIN_TORSO`, ...) contaminates measurements: pass them per command with `env VAR=... cmd` and measure with `env -u`.

## Project layout

```
models/fighter2d.xml          Generated MJCF model (scripts/build_model.py, copied from fightai)
scripts/env.py                Fighter2DEnv base class (copied from fightai)
scripts/locomotion_env.py     LocomotionEnv: stand / recover environment; train_locomotion.py trains the stand policy
scripts/floor_env.py          FloorEnv: lying starts, hand-over region (level), arrival rewards; train_floor.py trains the floor policies
scripts/handover_env.py       hand-over policy environment; train_handover.py trains it
scripts/controller.py         GetUpController: rule-based combination of the four policies (opt-in diagnostics for hand-over gates)
scripts/getup_env.py          GetUpEnv: the whole task as one 20 s environment; distill_controller.py / train_getup.py: single-network distillation and PPO
scripts/distill_stand.py, distill_floor.py   64x64 -> 256x256 distillation (behaviour cloning + value regression + DAgger)
scripts/analysis/             benchmark, controller_side_eval, funnel, supine_fall_diag, arrival_classifier, getup_eval, ... (diagnostics behind the docs)
scripts/*_notify.py, heartbeat.py, overnight_*.sh   unattended pipelines with KakaoTalk progress messages
scripts/render_demo.py        headless demo video (assets/)
checkpoints/                  trained models, autosaves, logs (gitignored)
docs/                         technical documentation (English summary, Korean experiment log and full history)
```
`build_model.py`, `env.py`, `view.py` and `kakao_notify.py` are copies of fightai's versions, not a shared dependency: a body change needs the same edit in both repos.

## Methodology note

SB3's training-log metrics (`fall_rate`, `ep_len_mean`, ...) are measured under stochastic rollouts and can move opposite to what matters, so every claim here is re-measured deterministically on fixed seeds per start side, compared on the same seeds, and given an error bar. The same seeds move +-4 points between runs and whole seed sets 4-5 points, so differences below ~3 points need 600+ paired starts. Recorded evaluation mistakes (leaked environment variables, a hand-over policy applied to the wrong side, starts that were not lying down) are listed in `CLAUDE.md` and the docs.

## Roadmap

Standing from any posture, held reliably -> walking (velocity tracking) -> transplant the policy into fightai's combat action/observation space (`transplant_locomotion.py`, still in fightai), each stage gated on the previous one actually working.

## License

MIT, see [`LICENSE`](LICENSE). `build_model.py`, `env.py`, `view.py` and `kakao_notify.py` are copies of files from the author's [`fightai`](https://github.com/SungyupKim/fightai) project.
