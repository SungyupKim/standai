# standai

Working notes for whoever (human or Claude) picks this project up cold. Full history and
reasoning: [`docs/stand_only_track.md`](docs/stand_only_track.md) (Korean, detailed) and
[`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) (English, condensed). This file is just
the "how do I actually run/resume/check things" operational layer.

## What's running right now

Check before assuming anything is idle or stale:
```bash
pgrep -af "train_locomotion.py|stand_progress_notify.py|watch_locomotion.py|train_floor.py|floor_progress_notify.py|watch_floor.py"
```
**State of play (2026-10-18) -- supersedes the 2026-10-17 block below where they conflict** (details: `docs/experiments_2026-10-11.md` 3.20, `docs/TECHNICAL_SUMMARY.md` 9):
- **Stand policy to use: `checkpoints/best_stand_v54.zip`** (= `v54` @20.3M, a copy of `autosave/ppo_stand_v54_lowlr_..._20300000_steps.zip`). Controller 20 s standing 41% (50:50 supine/prone) vs `v37` 29%. `v37`@96M is the old reference (pass it with `--stand`).
- Why it works: the stand trainer ran at the default lr 3e-4 with `approx_kl` ~0.09 and its hand-off catch jumped +-50 between checkpoints; `train_locomotion.py --learning-rate 5e-5` (new, also `--target-kl`) fixed that. Healthy: `approx_kl` 0.01-0.03.
- Start distribution env vars (all opt-in, default off): `STANDAI_HARSH_MIN_TORSO=0.7` (no lying harsh starts), `STANDAI_START_VEL_POOL/PROB/JOINT` (hand-off-like forward + joint velocity), `STANDAI_MIX_POOL/PROB/LAM_LO` (starts blended from two hand-off states; blends are easier than real hand-offs and mix supine with prone -- same-type blending is not done yet), `STANDAI_REPLAY_JITTER/VEL_JITTER/PITCH_JITTER`. v54's full command is in docs 3.20.
- **Always judge with `scripts/analysis/benchmark.py` (per side + 50:50 average).** The 80:20 supine/prone split of lying starts is an artefact of the collapse generator (hip target range), not a real fall distribution. Pick checkpoints by adjacent-checkpoint jitter and an independent eval set (`checkpoints/eval2_all.npz`, seeds 400+), not by one measurement.
- **Controller hand-over region: use level 0.7** (`FloorEnv(level=0.7)`; `benchmark.py --handover-level`, default 0.7 since 2026-10-18): 50:50 standing at 20 s 41% (level 1.0) -> 50% with v54 (docs 3.21).
- **2026-10-31: checkpoints are published at https://huggingface.co/SungyupKim/standai-getup** (staging folder `release/huggingface/`, git-ignored; model card = its `README.md`; upload with `.venv/bin/hf upload SungyupKim/standai-getup release/huggingface . --repo-type model`, the old `huggingface-cli` no longer exists in huggingface_hub >= 1). NEVER upload the whole `checkpoints/` folder: it contains `kakao_token.json`.
- **2026-10-26 (docs 3.24 addendum 13/14): BEST SYSTEM = ONE network, `checkpoints/best_getup_e2e1.zip` (= `getup_e2e1_final.zip`, 512x512, distilled from the controller, then 40M steps of end-to-end PPO with `scripts/train_getup.py`).** Fresh seeds 99100/99200/99300 (600/side, `scripts/analysis/getup_strict_eval.py`): supine 97.7%, prone 90.7%, **50:50 94.2%** (rule-based controller on the same seeds 70.8%). Shortcut checks done (root_x/root_ry ctrl 0, opponent moved 40 m: same, 150-500 N pushes: same, calm 5 s stand): `analysis/opponent_removed_eval.py`, `analysis/push_eval.py`. Drive it with `GetUpEnv` (`scripts/getup_env.py`), `render_demo.py --student <zip>`, evaluate with `getup_strict_eval.py --arms teacher <zip>` (paired, same seeds). The rule-based controller is now the TEACHER and the baseline. Size ablation DONE (`scripts/overnight_size_ablation.sh`, table in `checkpoints/analysis/size_ablation.out`, docs 3.24 addendum 15): 12M end-to-end steps, 50:50: 512x512 92%, 256x256 93%, 128x128 89%, 64x64 85%, rules 72% -> capacity is not the bottleneck down to 256; use 256x256 (`checkpoints/getup_e2e_256_final.zip`) as a lighter starting point. Nothing is training. Not tried: the same recipe from scratch at a large size (expected to fail), generalisation to other starts/bodies, walking from this network.
- **2026-10-25 evening (docs 3.24 addendum 13): EVALUATION BUG FIXED + system is ~73%, not ~68%.** `benchmark.task_controller` / `controller_side_eval.py` used to apply the prone-trained hand-over policy to SUPINE arrivals too when `--handover` was given (supine -5.5 points). Default is now `--ho-sides prone`; `side` = legacy. Corrected whole system on six fresh sets (1200/side): supine 61.8%, prone 84.4%, **50:50 73.1%**. The whole controller was distilled into one 512x512 network (`scripts/distill_controller.py`, `getup_env.py`, DAgger, `checkpoints/getup_student_distilled.zip`): 74.5% vs the rule-based 73.8% (paired, seeds 91000+/92000+). Running: end-to-end PPO `scripts/train_getup.py --out e2e1` (log `checkpoints/getup_train_e2e1.log`, trend `trend_e2e1.json`, autosaves `ppo_getup_e2e1_*`), PID check: `ps -eo pid,args | grep train_getup`. Judge it with `scripts/analysis/getup_eval.py --arms teacher <zip> --sets ... --n 200` (paired, same seeds).
- **2026-10-25 later (docs 3.24 addendum 12): supine bottleneck diagnosis.** Catch by the stand policy is almost determined by the arrival state (MLP AUC 0.946, `analysis/arrival_classifier.py`, labels `checkpoints/analysis/arrival_labels.npz`, classifier `checkpoints/arrival_clf.pt` via `analysis/train_arrival_clf.py`), but pushing the COM speed (`analysis/arrival_push_diag.py`) does not raise it, and a learned hand-over gate (`STANDAI_CLF_GATE=0.3`, opt-in, off by default) gives only +2-3 points (+41/1200). Nothing is training. Next idea not yet tried: distil the whole controller into one network and fine-tune end to end.
- **2026-10-25 (docs 3.24 addendum 11): `best_floor_supine_fdist.zip` stays arr1.** arr2 (stronger arrival reward) and catch1/catch2 (`STANDAI_CATCH_K`: stand-policy rollout outcome at the arrival as reward, `FloorEnv.catch_rollout`, opt-in) did NOT beat arr1 on 800 fresh starts (473 vs 501 / 488, +3.5 / +1.9 points, 1.5 / 0.8 sigma; threshold +4 points). Whole system ~68% (supine ~53-58%, prone ~80-86%). Nothing is training. Pitfalls: after a hard reboot the newest autosaves can be 0-byte files (pick the newest with size > 0 and `zipfile.is_zipfile`; `overnight_arr2_resume.sh` does); never `pkill -f`/`pgrep -f` a pattern that is in your own shell command line (kill explicit PIDs).
- **2026-10-24 later (docs 3.24 addendum 10): `checkpoints/best_floor_supine_fdist.zip` is now arr1** (fdist2 + 40M steps with `STANDAI_ARRIVAL_K=15`, a continuous reward at the success step for COM forward speed (target 0.45 m/s) and low pitch; `floor_env.py`, `scripts/overnight_floor_arrival.sh`; previous fdist2 = `floor_supine_fdist2.zip`). Supine 20 s standing 49.5% -> 60% on two fresh seed sets (+42/400), pooled four sets 56.5%; whole system on fresh seeds supine 53% / prone 80% / **50:50 ~66.6%**. Same-seed results still move +-4 points between runs and +-4-5 between seed sets. The reach rate of a floor checkpoint is NOT a selection criterion for 20 s standing (reach-picked checkpoint: no gain). Diagnostic tools: `analysis/funnel.py --supine --prone`, `analysis/supine_fall_diag.py` (state at the hand-over by outcome), `scripts/floor_trend_notify.py` (30-min reach + arrival speed/pitch for runs without a log table), opt-in `STANDAI_VX_GATE` hand-over delay in the controller (useless: <= +4 points). Running when this was written: arr2 (K 25, target 0.6 m/s, `overnight_arr2.sh`, `floor_train_arr2.log`, trend `trend_arr2.json`). PITFALL: `_prev_com_x` is used by `LocomotionEnv`; subclasses must use their own attribute name.
- **2026-10-24 (docs 3.24 addendum 9): `checkpoints/best_floor_supine_fdist.zip` is now fdist2 @128M** (+100M PPO steps; old = `floor_supine_fdist1.zip`): floor reach 70% -> 83-85%, but supine 20 s standing only ~45% -> ~49% (within noise). Fresh-seed whole system (seeds 53000+/54000+): supine 44.75%, prone 86%, **50:50 ~65.4%**. Re-training the stand policy on the new arrivals (big2, `stand_big2_last.zip`) did nothing in the controller -> `best_stand_big1.zip` stays. New: `scripts/floor_trend_notify.py` (30-min reach-rate trend for runs whose log has no tables), `make_handoff_pool.py --supine-only --level`, eval set `checkpoints/eval_fd2_all.npz`. Nothing is training.
- **2026-10-23 (docs 3.24 addendum 8): `checkpoints/best_handover_prone.zip` is now h4** (re-trained from `best_stand_big1`; prone 20 s standing 75% -> 83% on fresh seeds; old = `handover_prone_h3.zip`). Whole system ~65-67% (supine ~48-52%, prone ~83%). A supine hand-over policy (s3, big stand, 16,000 fdist arrivals `pool_fd_supine_all.npz`) did nothing again (99/200 vs 98 without). The same seeds vary +-4 points between runs. Nothing is training.
- **2026-10-22 (docs 3.24 addendum 6-7): current best system = `checkpoints/best_floor_supine_fdist.zip` + `best_prone8.zip` + `checkpoints/best_stand_big1.zip` + `best_handover_prone.zip`, level 0.7: 50:50 ~65%** (supine ~52%, prone ~77.5%, 400 starts per side over two fresh seed sets). The supine floor policy and the stand policy are 256x256 nets made by DISTILLING the old 64x64 policies (`scripts/distill_floor.py`, `scripts/distill_stand.py`: behaviour cloning of the mean action + value regression, DAgger) and fine-tuning with PPO (`overnight_floor_distill.sh`, `overnight_ladder.sh`); big floor policies trained from scratch all failed. Pass them explicitly (`--supine`, `--stand` in `benchmark.py`/`controller_side_eval.py`/`funnel.py`): the script defaults still point to v12 / v55b / the old demo. PITFALL: never `export` training-only start-distribution variables (STANDAI_HARSH_MIN_TORSO / START_VEL / MIX) in a shell that later runs benchmarks (it made 'lying' starts near-standing and inflated the results); use `env VAR=... cmd`. Kakao messages are tidied and split into <= 190-character parts by `kakao_notify.send_message`; `scripts/heartbeat.py` sends a 30-minute progress message. Nothing is training.
- **2026-10-19 latest (docs 3.24): a separate hand-over policy** `checkpoints/best_handover_prone.zip` (h3 @66.55M, warm-started from `best_stand_v55b.zip`, `scripts/train_handover.py`, env `scripts/handover_env.py`) sits between the floor and the stand policy for PRONE-side arrivals only (`GetUpController(..., handover_ckpt=..., handover_sides=('prone',))`, `benchmark.py --handover`, `analysis/funnel.py`). Whole system with v12 + prone8 + `best_stand_v55b.zip` + this (benchmark/funnel default stand policy is still `best_stand_v54.zip` in `benchmark.py`; pass `--stand checkpoints/best_stand_v55b.zip`; `funnel.py` defaults to v55b): 20 s standing = **50:50 62.5%** on untouched seeds 13000+ (200 per side: supine 49% / prone 76%; same seeds without the hand-over policy 54.8%, with stand v54 and no hand-over policy 47.8%). Supine hand-over policies (s1 `handover_supine_s1_unused.zip`, s2 `handover_supine_s2.zip` trained on level-0.3 arrivals) and a critic gate (`v_gate`) do NOT raise the supine side above ~48% (docs 3.24 addendum: the floor policy hands over back-leaning, unrecoverable states); the supine side is limited by the floor policy (27-30% never reach the region, arrival 5-7 s). A supine floor fine-tune with the hand-over critic as aux reward (sv1, `floor_supine_sv1.zip`) also gave nothing (docs 3.24 addendum 2). Nothing is training. Train: `STANDAI_NO_RECOVERY_GATE=1 python train_handover.py --init-from ../checkpoints/best_stand_v55b.zip --learning-rate 1e-4 --std-min 0.3 --first-index 16000` (from scratch it does not learn).
- **2026-10-19 later (docs 3.23): `checkpoints/best_stand_v55b.zip`** (v54 + COM-ahead ladder `--com-ladder --ladder-first-index 16000` + mirror + `STANDAI_STEP_K=300`, 30M steps) is slightly better than v54 (controller 50:50 54% vs 50%, prone catch 27-29% vs 26%, no regression) but within noise; prone hand-offs with COM > +0.10 m ahead of the front foot are still 0% caught. Nothing is training. Tools: `scripts/analysis/catch_by_com.py <ckpts>` (per-COM-bin catch, any eval set), `step_diag.py`.
- **2026-10-19 (docs 3.22): nothing is training.** Best system = v12 + prone8 + `best_stand_v54.zip`, controller hand-over level 0.7 (50:50 standing ~50%). The symmetrised stand policy (`scripts/sym.py`, `STANDAI_SYM_STAND=1`) is the same for v12 and fixes the left/right blind spot for any new floor policy (v12 hand-offs raise the LEFT leg; the stand policy catches 73% of them, 8% of the mirrored copies). Opt-in env vars: `STANDAI_FEET_GAP=0.15` (two-feet hand-off condition), `STANDAI_STAND_VALUE_CKPT/COEF/BONUS/SYM` (stand-policy critic as an auxiliary floor reward: tried on the supine policy (b3) and on prone8 (pv1), did NOT help -- b3 reached 40%, caught 28%, controller 10%). Viewer for the whole system: `DISPLAY=:1 .venv/bin/python scripts/watch_controller.py`. Supine from-scratch recipe that works: stage 1 with the DEFAULT lr 3e-4, loose target, supine starts only; low lr cannot learn to rise; speed penalty stalls the staged target.
- (superseded) Running: `ppo_floor_s4b` (supine floor, stage 2 of the original recipe: stage 1 `ppo_floor_s4a` = supine starts only, loose target, DEFAULT lr 3e-4; then stand-ready staged target, lr 1e-4; low lr from scratch did NOT learn to rise -- supine2/3), log `checkpoints/floor_train_s4b.log`.
- (superseded) Running: `ppo_floor_supine2` (supine floor policy from scratch, `--supine-frac 1.0 --level-start 0.0`, lr 1e-4; log `checkpoints/floor_train_supine2.log`, notifier `floor_progress_notify_supine2.log`). Paused (SIGSTOP): v46 (PIDs 288630..), v49 (PIDs 330512..). `ps -o pid,etime,cmd -C python` before launching anything.
- Hand-off pool for training/analysis: `checkpoints/pool_big_all.npz` (32,000 states, 16,000 supine + 16,000 prone, seeds 100000+). Eval sets: `eval_strict_all.npz` (300) and `eval2_all.npz` (300), each 150 supine + 150 prone.

**State of play (2026-10-17, previous):** three policies + a rule-based controller (details: `README.md` status table, `docs/experiments_2026-10-11.md` 3.17 and 3.18, `docs/TECHNICAL_SUMMARY.md` 6 and 7).
- Supine floor policy `ppo_floor_v12` (`autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip`), prone floor policy `ppo_floor_prone8` (`checkpoints/best_prone8.zip` = 86.5M; 77.45M / 82M / 84.25M / 97.8M are statistically the same).
- Stand policy: **`v37`@96M is the one to use** (127/300 of the strict hand-off evaluation set `checkpoints/eval_strict_all.npz`, calm gate ~50). `v38`, `v47` (hand-off pool fine-tuning) and `v39`-`v45` (from scratch) all FAILED to improve it
  (docs 3.18). Pool for training: `checkpoints/pool_strict_all.npz` (8,000 states, `scripts/make_strict_pool.py`).
- Best end-to-end system: `scripts/controller.py` (`GetUpController`) ~24% standing at 20 s (`scripts/analysis/controller_eval.py 1.0 1 200 1000`).
- **Running: `ppo_stand_v48_nofloor`** (PID 306780, notifier 306781; fresh, x1.5 body, `STANDAI_NO_RECOVERY_GATE=1`, `--no-knee-floor`, `--std-max 0.4`, no pool; log `checkpoints/locomotion_train_stand_v48.log`). The training-time knee floor (-50 -> -80 deg, model range -140) was a crutch; without it a fresh run leaves gate 0 for the first time (alarm gate 0.37-0.47 by 10M). Benchmark @16.8M is still below v37 (catch 88/300 vs 127, calm 41 vs 51, controller 20% vs 24%): benchmark every ~10M steps; it replaces v37 only if catch > 127/300.
  `STANDAI_NO_RECOVERY_GATE=1 nice -n 10 .venv/bin/python -I scripts/analysis/benchmark.py --stand <ckpt> --tag <name>` (~4 min; whole system in one table, defaults = v12 / prone8 / v37@96M).
  **Paused (SIGSTOP):** `ppo_stand_v46_v21like` (PIDs 288630 + forkserver children, notifier 288631): resume with `kill -CONT <pids>`. Next steps after v48 works: hand-off pool stage on top of it (performance-gated, failure-weighted), then supine floor retrain with `train_floor.py --supine-frac 1.0` (prone recipe).
  If it was paused: `kill -CONT <pid and its forkserver children>`. ALWAYS `ps -o pid,etime,cmd -C python` (with env: `tr '\0' '\n' < /proc/<pid>/environ | grep STANDAI`) before launching a new run: a duplicate launch already happened once.
- Floor runs use `--no-down-timeout` (env.py's 5 s "stayed down" KO off), `--max-steps 450`, `--standready`; `--prone-frac 1.0` = prone starts only; the staged target is `--level-start 0.0` (works only from a network that already gets high; from scratch it never left level 0).
- Pick checkpoints by re-measuring candidates on the SAME 400 seeds (`scripts/analysis/eval_ckpt.py`); state the "held" criterion every time (never below 0.7 m for 3 s vs torso >= 0.7 m at 3 s differ hugely for the floor policies).
- **Do not trust the trainer's `fall_rate` / `ep_len_mean` for stand runs:** the training-time knee floor (-50 -> -80 deg in calm episodes) is a rigid support (KO 16/16 without it, 10/16 with it, for a policy whose real gate is 0). Use the fixed-seed gate and
  `scripts/analysis/stand_posture_diag.py` / `stand_longrun_diag.py`. Use the system `.venv/bin/python` (system python3 has no numpy).
**Body spec changed (2026-10-11): limb actuator gears are x1.5 by default** (`BODY_GEAR_SCALE` in
`locomotion_env.py`, hip 270 / knee 90 / ankle 60; root_x/root_ry untouched). Reason: a closed-loop MPC
oracle stands from only 2/6 lying starts at x1 but 5/6 at x1.5 (same as x2), and an unchanged policy
gains on crouched calm starts at x1.5 without losing upright ones. **Every result recorded before v28
(including all "baselines" below: calm 28-40 / full 17-23 / harsh 5-7) is at x1.** To reproduce an old
number run the script with `STANDAI_BODY_GEAR=1`. The training/gate/notifier/viewer all pick the body up
from the env automatically; never compare gate numbers across body specs.
v28 (pure PPO, fixed x1.5 body, calm_frac 0.6) went 41 -> ~50 calm in ~4M steps and then plateaued;
no harsh start that begins below 0.6 m torso height has ever ended upright (the "harsh improvement" came
from starts at 0.62-0.67 m). The policy tumbles on the ground and swings its legs; `knee_avoid` paid for
raised knees. v29 (v28 @106.3M + `--knee-gate`: knee_avoid faded out by head height, 0 below 0.5 m / full
above 0.9 m) ran only ~1M steps before being replaced -- no verdict on the gate alone.
v30 was a calm-only restart (from v21 @82.425M, body x1.5, `--knee-gate --head-progress 300`, 7M steps, calm_frac
1.0 -> 0.69): rolling dropped (1.4 -> 0.95 turns/episode) but calm slipped below the control (127/300 vs 136/300)
and no harsh start below 0.6 m ended upright. `--head-progress` (potential-based) nets to zero over a rise-and-fall,
so it gives almost no net incentive -- that was the mistake in calling it "drastic". Leg lifting while lying is NOT
a harsh-learned habit: v21 @82.4M (never saw harsh) already swings its legs and rolls 1.5 turns on the floor
(calm-learned reflex).
**v31 = v30 @89.5M + `--head-record 400`** (replaces head-progress; ran only ~0.3M steps before v32): pays K * (new best normalised head height -
previous best) once per height per episode, only with a foot on the floor; not cancelled by falling back, bounded by
K*(1-start height) ~ 320, 0 on calm starts. `--knee-gate` kept; `--calm-already-relaxed --calm-end-frac 0.5` so more
episodes are floor starts. Probe: random flailing already earns ~31/episode and v30 ~28 (84% of floor episodes pay),
so the low end is easy to reach and what separates policies is how high the head gets.
**v32 = v31 + correlated exploration noise** (`ColoredNoiseCallback`, ran ~0.8M steps): rollout actions use an AR(1)
noise (rho 0.9, per-step marginal = the policy's own std) instead of SB3's independent per-step noise. Side effect to
avoid: its `--floor-noise-scale 1.5` (noise 1.5x wider than the policy's on the floor) is an off-policy sample the policy
learns to avoid, so its std shrank 0.50 -> 0.46 (and to 0.345 within minutes of v33). Keep that at 1.0.
Key finding behind v33/v34 (`scripts/analysis/random_explore.py`): noise added AROUND the policy's mean stays anchored to
what the policy does (leg lifting) -- the policy lifts the head above 0.5 m on 6% of floor starts, but policy-FREE random
+-1 limb patterns held 10 steps do it on 77% (0.9 m: 11%, one genuine upright of 126; hold 25: 52%/6%; hold 50: 29%/3%;
smooth random AR(1) noise only 6-31%). The learned policy is WORSE than random bang-bang at raising the head.
v33 (stored the +-1 burst as the policy's action) diverged immediately: approx_kl 2e6, clip_fraction 0.95, std collapse,
then NaN -- a +-1 pattern is far in the tails of the policy's Gaussian, so PPO's importance ratios explode. Lesson: never
store off-support actions in the PPO buffer.
**v34 = v31 @89.85M + env-level floor bursts** (`--floor-burst-rate 0.1 --floor-burst-hold 10`): on envs with torso < 0.5 m,
a random +-1 limb pattern held 10 steps is executed by the ENV (via a `step_async` override) while the buffer keeps the
policy's own on-policy sample, i.e. the burst is an environment disturbance that puts the body in raised-head states the
policy then continues from (during a burst its stored action is uncorrelated with the outcome: noise, zero mean gradient).
~half of floor time is in bursts. Healthy at launch: approx_kl 0.04-0.05, std 0.51, `r_head_record` 7 -> 28-34/episode.
Log keys: `noise/floor_env_frac`, `noise/burst_active_frac`, `r_head_record`.
v34 result (6.7M steps, 90.0M -> 97.2M): calm held (gate 43-54, 300-seed 150/300), but the policy ALONE (no bursts) did not
move on floor starts (head>0.5 m: 5% -> 7%, >0.9 m: 0%, upright 0/126), its action std kept shrinking (0.51 -> 0.32), rolling
went UP (1.42 turns, 17% inverted) and value_loss doubled. Bursts raise the head during training (r_head_record 7 -> 25-34)
but nothing transfers to the policy by itself: the burst's motion is not in the PPO buffer.
**v35 = v34 @97.2M + `--burst-archive-frac 0.25 --std-min 0.45 --ent-coef 0.002`** (two ways to make it stick):
(1) `BurstArchiveCallback`: the env reports states reached from a floor start with head 0.45-0.9 m, a foot down and the
torso not inverted (`info["archive_state"]`, at most one per 10 steps); they are kept in 4 head-height bins and 25% of ALL
resets start from them (balanced across bins, via set_replay_pool) -- pure RL, the policy practises "continue from a
raised pose". (2) std floor 0.45 (lifted at load: 0.315 -> 0.450) + a small entropy bonus to stop the std shrinking.
Not done (user's call): self-imitation of successful bursts (averaging random patterns may give ~0 mean action).
Log keys: `archive/size`, `archive/bin*`, `archive/success_replayed`, `noise/burst_active_frac`, `r_head_record`.
v35 ran only ~0.7M steps (policy alone: head>0.5 m 8%, >0.9 m 0%, upright 0/126, std held at 0.45) before being replaced.
**v36 = the same flag set restarted from the calm-only v21 @82.425M** (warm-start chain reset to 1: v21 -> v36) with
`--calm-already-relaxed --calm-end-frac 0.5` from step 0. Why no calm-only warm-up phase: bursts, the head-record bonus and
the archive all need FLOOR episodes, so a calm_frac that starts at 1.0 produces none of them at first; with 0.5 from the start
r_head_record is 29-33/episode within minutes. v21 already has the calm skill (43 gate on the x1.5 body) and the leg-swing
reflex too (so the restart does not remove that habit; it removes the std drift / extra rolling accumulated in v32-v35).
v36 ran ~4M steps (policy alone: head>0.5 m 4%, >0.9 m 0%, upright 0/126, same as its v21 start; archive-started episodes held a
3 s stand 1.3% of the time). Chain probe (`scripts/analysis/chain_probe.py`, 123 mid-height states, 5 s): from head 0.6-0.9 m the
policy gets the head past 0.9 m 30% of the time, but 16 of the 19 episodes that did fall back (median 0.56 s above 0.7 m torso
height; the gate needs 3 s) -> the weakest link is balance right after standing up, not the rise itself; from head 0.45-0.6 m the
policy (6%) is WORSE than random bursts (15%).
**v37 = v36 @86.45M with a FRONTIER ladder archive** (same flags; the change is in the env/callback): (1) rungs now span
head 0.25 -> 1.10 m in 7 bins (0.25/0.35/0.45/0.55/0.65/0.75/0.90/1.10): the lowest rung is lying with the head just off the floor
(before, nothing below 0.45 m was stored, so the lying->0.45 step had no rung) and the top rung is "just stood up, still moving".
(2) Frontier rule (Go-Explore style): a state is stored only if the head is >= 0.08 m above where THAT episode started, and archive-
started episodes contribute too, so each rung seeds the next. Verified: round 2 (started from round-1 states) produced states +0.18 m
above their own starts on average, max head 0.95 m; calm starts contribute nothing. Bursts still apply only while torso < 0.5 m,
so the upper rungs are practised by the policy alone.
Judge v37 by (all measured on the policy ALONE): harsh starts below 0.6 m ending upright (docs 3.8; 0/160 for every checkpoint so
far), `random_explore.py <ckpt>` policy row (head>0.5 / >0.9 m on floor starts), `chain_probe.py <ckpt>` (does the rise continue from
mid-height states and, above all, does it END upright: 3/123 now), `archive/success_replayed` and the `archive/bin*` sizes (the top
rungs filling is the sign the ladder is working), tumble count, std staying ~0.45, calm / upright-start (~77-85%) holding.
Abandoned: v22/v23 (plateau at x1), v25 (`--getup-reward`, 36M steps), v26 (`--replay-curriculum`, 9M steps), v27
(`--gear-curriculum`, 1.8M steps), v28 (plateau ~50 calm, 0 floor get-ups), v29 (~1M steps), v30 (7M steps).
All of `--getup-reward` / `--ground-shaping` / `--replay-curriculum` / `--gear-curriculum` / `--knee-gate` /
`--head-progress` / `--head-record` / `--colored-noise-rho` / `--floor-burst-rate` / `--burst-archive-frac` / `--std-min` are opt-in. Full write-up: `docs/experiments_2026-10-11.md`.

**v38 = v37 @96M restarted with hand-off starts (2026-10-12).** Why: v37's calm gate (100 seeds) fell 49-51 (90-99M) -> 45/41/47/38 (102-112M) while
the training log looked fine, and the stand policy catches only ~7% of the states the floor policy hands over (5-9% for every checkpoint, so the
restart point was chosen by calm gate: the last point before the decline, 96M). New flags: `--handoff-pool ../checkpoints/handoff_pool_v4.npz
--handoff-frac 0.4` (`HandoffPoolCallback`; pool = 1500 states from `scripts/make_handoff_pool.py`, seeds 1000+, disjoint from the 150-state eval set
`handoff_eval_v4.npz` = seeds 0.., deterministic floor policy). Bursts / archive / head-record / colored noise / entropy bonus are OFF (floor tooling).
The pool is FIXED (not refreshed) -- if hand-off catch rises on the pool's states but not on the eval set, it is memorising; next step then is a
failure-weighted refresh. Baseline at launch: hand-off upright 11/150, calm gate 51/100. Judge v38 by the notifier's "hand-off catch" line and the
calm gate, not the training log.
(v38 FINISHED 2026-10-12 at 136M with no gain; the command below is kept only to reproduce it.) Resume v38: `cd scripts; ../.venv/bin/python train_locomotion.py --n-envs 8 --timesteps 40000000 --stand-only --out ppo_stand_v38_handoff
--height-start-scale 6.0 --height-end-scale 15.0 --height-relax-frac 0.3 --height-already-relaxed --curriculum-fall-gate 0.65
--curriculum-streak-gate 350 --calm-already-relaxed --calm-end-frac 0.8 --handoff-pool ../checkpoints/handoff_pool_v4.npz --handoff-frac 0.4
--ent-coef 0.0 --init-from "$(ls -t ../checkpoints/autosave/ppo_stand_v38_handoff_*_steps.zip | head -1)" >> ../checkpoints/locomotion_train_stand_v38.log 2>&1 &`
(notifier: `stand_progress_notify.py --log ..._v38.log --ckpt-glob '../checkpoints/autosave/ppo_stand_v38_handoff_*_steps.zip' --total-timesteps 40000000
--baseline-steps 96000000 --handoff-eval ../checkpoints/handoff_eval_v4.npz`). Chain is now v21 -> v36 -> v37 -> v38 (4 warm starts; CLAUDE.md warns at 5+).

**Floor get-up is now a SEPARATE policy (2026-10-12)** -- `scripts/floor_env.py` (`FloorEnv`), `train_floor.py`,
`floor_progress_notify.py`, `watch_floor.py`. Why: the single stand policy never got off the floor (0/160 genuine uprights
below 0.6 m, calm-learned leg-swing reflex); a fresh network on floor-only episodes has no such habit. Every episode starts
lying (torso < 0.5 m); reward is only head-record + small head/torso height terms + a success bonus. **Success = REACH the
hand-off pose** (torso >= 0.6 m, head >= 0.9 m, foot down, |pitch| < 60 deg), episode ends there; holding it is the STAND
policy's job. (v3 required holding 20 steps -- that made the floor policy do balance too and its success stayed 0 while 18%
of seeds reached the pose; v4 went back to reach.) Hand-off states are reported in `info["handoff_state"]`.
Combine with the stand policy by a state switch (floor policy until hand-off, stand policy after; hysteresis); the switch
and the hand-off boundary are NOT built yet (idea: a small classifier "does the stand policy hold 3 s from this state").
Findings: env-level bursts HURT a fresh floor policy (v1 w/ bursts: head>0.5 m 20% policy-alone at 1.3M vs v2 w/o bursts 72% at
0.63M) -- use `--floor-burst-rate 0`. Measured catch rate: floor policy reaches hand-off in 5% (v2) / 18% (v3) of 100 seeds and
the stand policy v37 then holds 3 s in only 1/5 / 2/18 of those -- the stand policy must be trained on hand-off states.
Floor action std is ~0.64 on every joint (stand policy 0.47) and 65% of deterministic actions are saturated (|a| > 0.9): getting
up on this body needs near-max torque and wide moves; std is a state-independent parameter so one policy cannot be wide on the
floor and narrow when balancing (one more reason for the split). KL stays ~0.3-0.4 / clip_fraction ~0.75 (candidate fix if
success stalls: lower learning rate, entropy bonus off; `train_floor.py` has no --learning-rate flag yet).
Lineage: v1 (bursts) -> v2 (no bursts, loose success) -> v3 (hold success) -> v4 (reach success, from v3 @5.925M, 8 envs):
`ppo_floor_v4`, log `checkpoints/floor_train_v4.log`, notifier `floor_progress_notify.py` (policy-alone eval every 30 min).
Resume: `cd scripts; ../.venv/bin/python train_floor.py --n-envs 8 --timesteps 20000000 --out ppo_floor_v4
--floor-burst-rate 0.0 --init-from "$(ls -t ../checkpoints/autosave/ppo_floor_v4_*_steps.zip | head -1)" >> ../checkpoints/floor_train_v4.log 2>&1 &`.
Watch: `DISPLAY=:1 ../.venv/bin/python watch_floor.py <ckpt> [--only-success]` (keeps running 3 s after a success).
Watching shows supine starts try a bent-leg get-up and prone starts a push-up.
**FINAL: v4 finished at 25.9M steps: reaches the hand-off pose in 80/100 deterministic starts (median 1.8 s); with v37@99M catching it, 8/100 end-to-end upright (3/100 with v37@120M).**
**Earlier, v4 @10.2M, policy alone (100 seeds): reaches the hand-off pose 38/100 (24% -> 34% -> 38% over 7.6M -> 10.2M); stand policy v37 then
holds 3 s in 3/38 -> 3/100 end-to-end upright from lying starts (was 0/160).** Reach is 38/82 on one start side (sign of initial
torso pitch) and 0/18 on the other (face-up/down mapping unverified). Next: train the stand policy from `info["handoff_state"]`
states, build the switch, lower the floor learning rate (approx_kl ~0.45, clip_fraction ~0.78).

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
  limbs -- no single group reproduces it (waist/arms untested; n=6 is noisy). v28 result: calm gate 47-54,
  300-seed calm 152-154. CORRECT CONTROL = v21 @82.4M on the x1.5 body: calm 43 / full 24 / harsh 5,
  136/300 calm, 6/197 harsh starts end upright (all starts at 0.6-0.7 m; 0/160 below 0.6 m). So v28's gain over
  control is small (+16-18/300) and no harsh start below 0.6 m has ever ended upright.
- recovery_gate (knee_straight/torso_upright off when low) is fine: v22 (no gate) vs v23 (gate) from
  the same init gave the same gates. Do not revert it.
- Literature (Tao+ 2022 "Learning to Get Up", HumanUP 2025): curriculum + staged difficulty; HumanUP
  used ~5B steps, i.e. ~100x what we run. Strong-body-first is what v27 tries;
  a canonical-pose ladder (squat/kneel/sit/lie) is not yet tried.
- Judge v37 by the gate snippet below (x1.5 body, the env default). v28 reference on the same body: calm
  47-54 / full 23-29 / harsh 6-9 (100 seeds); its init checkpoint was 41 / 22 / 5. Also the 300-seed knee-bin
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

ckpt = max(glob.glob('checkpoints/autosave/ppo_stand_v37_ladder_*_steps.zip'), key=os.path.getmtime)
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
LATEST=$(ls -t ../checkpoints/autosave/ppo_stand_v37_ladder_*_steps.zip | head -1)
nohup ../.venv/bin/python train_locomotion.py --n-envs 8 --timesteps 60000000 --stand-only \
  --out ppo_stand_v37_ladder \
  --height-start-scale 6.0 --height-end-scale 15.0 --height-relax-frac 0.3 --height-already-relaxed \
  --curriculum-fall-gate 0.65 --curriculum-streak-gate 350 \
  --calm-already-relaxed --calm-end-frac 0.5 --knee-gate --head-record 400 \
  --colored-noise-rho 0.9 --floor-noise-scale 1.0 --floor-burst-rate 0.1 --floor-burst-hold 10 \
  --burst-archive-frac 0.25 --std-min 0.45 --ent-coef 0.002 \
  --init-from "$LATEST" \
  >> ../checkpoints/locomotion_train_stand_v37.log 2>&1 &
disown

nohup ../.venv/bin/python stand_progress_notify.py --log ../checkpoints/locomotion_train_stand_v37.log \
  --ckpt-glob '../checkpoints/autosave/ppo_stand_v37_ladder_*_steps.zip' \
  --total-timesteps 60000000 --interval-minutes 30 >> ../checkpoints/stand_progress_notify_v37.log 2>&1 &
disown
```
`--calm-already-relaxed` matters just as much: without it a restart resets `calm_frac` to 1.0 (all
calm) and re-anneals down, silently cutting harsh exposure for ~9M steps. `--calm-end-frac` sets the
harsh/calm target (0.6 for v28/v29; the env default is `CALM_EPISODE_FRAC` = 0.45). The x1.5 body is the env default (no flag); `--knee-gate --head-record 400 --colored-noise-rho 0.9 --floor-noise-scale 1.0 --floor-burst-rate 0.1 --floor-burst-hold 10 --burst-archive-frac 0.25 --std-min 0.45 --ent-coef 0.002` must be passed on every v37 resume (the archive is rebuilt from scratch after a restart; it is not saved).
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
ckpt=$(ls -t ../checkpoints/autosave/ppo_stand_v37_ladder_*_steps.zip | head -1)
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
