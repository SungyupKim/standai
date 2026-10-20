# standai — Technical Summary (condensed, English)

*Full detailed history (Korean) lives in [`stand_only_track.md`](stand_only_track.md). This is a
shorter English companion covering the architecture and the key findings, not every iteration.*

## 1. What this is

A 2D ragdoll fighter simulated in MuJoCo, trained with PPO (Stable-Baselines3). The original goal
was a self-play combat league; partway through, a harder question surfaced — can this body learn
to *stand*, from an arbitrary starting posture, at all? — and the project pivoted to answer that
rigorously before returning to combat.

- Physics: MuJoCo, contact forces via `mj_contactForce`, 2D sagittal-plane ragdoll (two mirrored
  fighters, `a_`/`b_` prefixed bodies)
- RL: Gymnasium custom env + SB3 PPO, `SubprocVecEnv` with 8 parallel workers
- Combat mode: self-play with periodically-frozen opponent snapshots, mirrored observations
- Stand-only mode (`LocomotionEnv(stand_only=True)`): no opponent, learn balance/recovery alone

## 2. Methodology (the part that mattered most)

The single biggest lesson across this project: **training-log aggregates (`fall_rate`,
`ep_len_mean`, rolling-window means) are not trustworthy on their own.** Measured repeatedly that a
training-log metric can look like steady improvement while the actual behavior of interest is flat
at zero. The fix, applied consistently from here on: a **fixed-seed, deterministic-action gate** —
reset with seeds `0..N-1`, run the current policy with `deterministic=True`, and measure the thing
you actually care about directly (e.g. "does the torso stay above a height threshold for 3
continuous seconds"). Every claimed result in this project is backed by that kind of direct
measurement, not a log line.

Related habits that came out of repeated mistakes:
- **Audit every legacy reward term when entering a new mode.** Two separate bugs (a "stance
  width" reward and a "getting up" reward) kept paying out during stand-only training because
  they were designed for combat and never explicitly disabled for the new mode — same root cause
  twice.
- **Pure reward-constant changes should continue from a checkpoint (`--init-from`), not restart
  from scratch.** Body/XML changes and observation-space changes do require scratch retraining;
  reward-coefficient tweaks don't.
- **Curriculum transitions should be performance-gated, not time-gated.** A fixed-step schedule
  can't tell whether the easy phase actually finished before harder cases get mixed in.

## 3. The stand-only investigation, condensed

**Can the body physically stand?** Computed `qfrc_bias` (the torque needed to hold a pose
statically) at a hand-set fully-upright configuration: each joint needs only **1–5% of its max
torque** to hold a straight-legged stand. Actuator strength is not the bottleneck.

**Then why does it keep crouching?** A hand-coded capture-point PD controller (no RL) was used to
separate "body can't do it" from "policy hasn't learned it." Scanning actuator gear with that
controller (not the reward formula) found the real lever: tripling ankle gear and doubling hip
gear raised survival from 3% to 17%, while boosting hip alone made it *worse*. This matched a
biomechanics audit: this body's ankle:hip torque ratio was ~1:6.4 vs. a human's ~1:2 — the ankle
was under-strengthed for the fine balance corrections it's supposed to provide. Gear was retuned
(hip 90→180, ankle 14→40) to match.

**Body proportions were also off.** Measuring actual body-part masses against Winter's
anthropometric tables: head was 12.3% of body mass (human ~8%), one leg was 23.5% (human ~16%),
trunk was 33.6% (human ~50%). Fixed via density adjustments on existing geometry (head/foot/leg
density down, trunk density up) rather than resizing capsules, landing within ~1pp of the human
reference ratios at the same total body mass. The hip joint's range of motion was also found to be
symmetric (−105°..105°) when every other joint sharing the same axis convention had already been
given an anatomically asymmetric range (hips can flex to ~120° but only extend ~10–20° past
neutral in reality) — fixed to −20°..120°.

**Two reward leaks, same shape.** A "stance width" reward (pays for spreading the feet, meant to
keep a fighter from being easily swept in combat) was never disabled for stand-only and was
rewarding leg-splitting for free. A "recovery" reward (meant to pay for genuinely getting up off
the ground) was gated against a near-fully-upright reference height, so it stayed ~25% open even
in a relaxed, already-passing stance — rewarding small vertical bobbing ("tap-dancing") instead of
holding still. Both were gated off above the fall-height threshold once found.

**A reward for the thing the gate actually measures.** Training-log `fall_rate` kept improving
while the real 3-second gate stayed at 0% because the knockdown-recovery window (5 seconds) made
"fall, recover, repeat" nearly free. Added `stand_streak`: reward scales with consecutive steps
spent above the fall-height threshold (saturating at 150 steps = 3s), resetting hard to zero on any
dip — directly incentivizing the thing being measured, not a proxy for it.

**A missing observation feature.** Even after all of the above, training plateaued for 20M+ steps.
The hand-coded controller's edge was that it was handed the centre-of-mass position/velocity
relative to the feet directly; the RL policy only had raw joint angles/velocities and had to
rederive that relationship (forward kinematics) from scratch, inside episodes that often ended in
well under a second. Added the capture-point-style features (`com_rel_x`, `com_rel_z`,
`com_vel_x`) directly to the observation (66 → 70 dims).

**A curriculum that asked for two skills at once.** "Hold a stand" and "get up from a collapse"
are different skills; the existing calm/harsh episode-start mix (tuned for the combat league) had
never been re-examined for the stand-only goal. Split into a gated two-stage curriculum: phase 1
trains calm (near-standing) starts exclusively until `fall_rate ≤ 0.1` and the longest continuous
stand streak reaches `≥500` steps (10s) over 200 episodes; only then does phase 2 phase in
harsh/collapse starts.

**A knee that wouldn't straighten, twice.** Watching trained checkpoints directly: both knees sat
pinned near their flexion limit even in a genuinely stable (near-zero-velocity) equilibrium — the
existing "brace" reward for straightening knees only fired during an active fast fall, not while
already resting low. Added an always-on knee-straightness reward. First version averaged both
knees, which a sharp catch (from direct observation, not a metric) revealed was being gamed by
straightening just one leg while leaving the other fully bent — fixed to use the minimum of the two
knees, matching a fix an existing reward term (`knee_avoid`) had already made for the identical
problem.

**Discovering a stale override.** While raising the head-height reward scale, printing the value
before vs. after the intended change caught that a much older experiment had left a module-level
override in place (ambient baseline was actually 6.0, not env.py's bare default of 3.0) — the
intended "3.0→5.0" edit would have been a silent regression. Caught by verifying the actual runtime
value instead of trusting a stale comment.

**First real success.** With the height-reward scale raised and ramped as a proper scaffold
(rather than a one-off manual edit each time), a fixed-seed re-measurement with 100 seeds showed
the deterministic policy genuinely achieving the 3-second survival bar — **1/100**, the first
non-zero result across the entire stand-only effort. Still far from solved, but the central
open question — "is this even achievable?" — now has a direct, measured "yes."

**A measurement bug retracted a conclusion.** Raising the knee-straightness reward appeared to
have zero effect on actual knee angle — until it turned out the manual diagnostic script wasn't
applying the knee-angle floor scaffold (−80° during calm episodes) that training actually used.
Re-measured with the matching floor applied: single-episode torso height snapshots jumped from
0.10–0.43 to 0.76–0.96. The "no effect" claim was wrong; the real fixed-seed gate result at that
checkpoint was unaffected either way. A related bug was found the same night: the interactive
viewer script had never actually been passing `stand_only=True`, so every manual "does this look
right" check had secretly been running with the torso-pitch assist force re-enabled — easier
conditions than what training (and the gate) ever sees.

**The torso visibly tumbles — a missing control skill, not a missing actuator.** Torso pitch has
no directly-driven actuator; combat relies on an always-on external PD that's deliberately zeroed
for stand-only. Watching the (bug-fixed) viewer run unassisted showed the torso rotating freely
with no resistance. Reasoned this is a learned-skill gap rather than a structural one — humans
don't have a direct torso muscle either, they control trunk orientation indirectly through
hip/ankle ground-reaction moments — and added an angle-based `torso_upright` reward using the
existing action space.

**Too much accumulated warm-starting, tested directly.** After five-plus sequential
`--init-from` reward-constant changes chained together, the resulting policy's behavior looked
qualitatively "off" in the viewer — raising a concern that repeated warm starts had destabilized
PPO's value function. Rather than guess, abandoned the chain and restarted completely from
scratch (run `v21`) with every currently-settled setting (body proportions, the full 70-dim
observation space, the gated curriculum, all current reward scales) baked in from step 0. This
doubled as a direct test of the accumulation hypothesis.

**The scratch restart won, clearly.** `v21` reached its first non-zero gate result (5/100) in
under 12M steps — the old warm-started lineage had taken 33M+ steps to reach 1/100. The lead held
up as training continued: by 53–55M steps, `v21` was clearing **30–40%** on calm-start fixed-seed
gates and **17–23%** on the full calm+harsh distribution. This doesn't overturn the "continue
reward-constant changes via `--init-from`" rule from §2 outright — it looks more like that rule
has a limit (somewhere under five sequential changes) rather than being wrong in general.

## 4. Where it stands now

> **Update 2026-10-12** -- superseded in part; see §5 and the full write-up in
> [`experiments_2026-10-11.md`](experiments_2026-10-11.md). The single stand policy (now `v37`,
> ~102M steps, x1.5 limb-torque body) plateaued at calm ~50-57% / full ~23-30% / harsh ~5%, and no
> checkpoint ever got up from a start below 0.6 m torso height (0/160). The task was therefore split
> into a stand policy and a separate floor get-up policy (§5). "Physical capability is confirmed
> sufficient" below applies to *holding* a stand, not to getting up. Numbers in the bullets below are
> from the original-gear body and are not comparable.

- Standing from an arbitrary posture, held for 3 full seconds, deterministic policy, fixed seeds:
  **~30–40%** on near-standing starts, **~17–23%** on the full calm+harsh distribution, as of the
  current run (`v21`) at ~55M steps — up from a flat 0% across roughly twenty prior iterations and
  a first breakthrough of ~1%.
- Training-log `fall_rate` staying stubbornly high (0.55–0.75) while the real gate keeps climbing
  is read as a *healthy* sign here, not a contradiction: earlier iterations had low fall_rate with
  zero real success (a safe, low-crouch local optimum); `v21` appears to be genuinely attempting
  the harder, correct target (a tall, upright stand) and succeeding at it some of the time, rather
  than settling for the easy wrong answer.
- The curriculum is still holding at 100% calm (near-standing) starts — the performance gate to
  start mixing in harsh/collapsed starts (`fall_rate ≤ 0.1` and max streak ≥ 10s, sustained over
  200 episodes) hasn't been cleared yet.
- Physical capability is confirmed sufficient; remaining work is about reward/curriculum shaping
  finding the easy, low-torque equilibrium reliably rather than settling for a low, stable crouch.
- This track was split out of the original combat repo (`fightai`) into its own repo (`standai`)
  once it was clear it needed independent iteration speed; `build_model.py`/`env.py` are copied
  rather than shared, so a body change has to be applied in both places by hand.
- Walking and the combat league are intentionally on hold until standing is solid — the project's
  own working hypothesis is that sequencing "stand" → "walk" → "fight" as separate, gated curriculum
  stages will go faster than tackling them together, for the same reason splitting "hold a stand"
  from "get up from a collapse" did.

## 5. Update 2026-10-12: a separate floor get-up policy

**Why split.** Everything tried on the single policy left floor behaviour unchanged: reward
shaping, failed-start replay, a 1.5x stronger body (helped calm only), correlated exploration
noise, env-level "bursts" (random +-1 limb patterns executed by the environment, never stored in
the PPO buffer) and a frontier-ladder reset archive (Go-Explore style). Policy-alone, head above
0.5 m on only 4-8% of floor starts, versus 77% for policy-free random bang-bang patterns. The
floor behaviour is a leg-swinging reflex learned during calm training, and a single state-independent
action std cannot be wide on the floor and narrow while balancing.

**Design.** `FloorEnv` (`scripts/floor_env.py`) starts every episode lying (torso < 0.5 m) and
ends it on reaching the *hand-off pose*: torso >= 0.6 m, head >= 0.9 m, a foot on the floor,
|pitch| < 60 deg. Reward is deliberately plain: a head-height record bonus (pays only for a new
best height, with a foot down), small head/torso height terms, and a success bonus; no balance
terms. A fresh network is trained with plain PPO (`train_floor.py`, 8 envs). The reached state is
reported in `info["handoff_state"]` so the stand policy can later practise from exactly the states
the floor policy hands over. Holding the pose afterwards is the stand policy's job.

**Results** (deterministic policy alone, fixed seeds, not training-log numbers):

| Run | Steps | Head > 0.5 m | Reaches hand-off |
|---|---|---|---|
| v1, with bursts | 1.3M | 20% | -- |
| v2, no bursts, loose success | 0.6M | 72% | -- |
| v4, reach success | 7.6M / 9.3M / 10.2M | 92% (9.3M) | 24% / 34% / 38% (100 seeds at 10.2M) |

End to end (floor policy until hand-off, then the stand policy `v37` for 3 s, 100 seeds): reached
38, still upright after 3 s **3** -> **3/100** from lying starts, where the single policy had 0/160.
The stand policy has never trained on hand-off states; that is the next step. The floor policy
reached the pose from 38/82 starts of one side and 0/18 of the other (split by the sign of the
initial torso pitch; the face-up/face-down mapping is unverified).

**Lessons recorded the hard way.**
- Bursts help a pretrained policy's *training data* but hurt a fresh policy: during a burst the
  stored action is uncorrelated with the outcome, so up to half of floor time carries no gradient,
  and the training-log metrics are inflated (46% vs 6% policy alone at 0.2M steps).
- A success defined as "head >= 0.9 m once" produced a pop-up (head up, torso back to 0.3-0.5 m
  within 3 s). A first fix required holding the pose 20 steps, which made the floor policy also
  responsible for balance and cut its measured success to 0% even though 18% of seeds reached the
  pose; the final definition is reach-and-hand-off, with holding measured separately by
  actually handing over to the stand policy.
- The check that first exposed the pop-up continued the *floor* policy after the success, which
  says nothing about hand-off quality. The hand-off has to be tested with the policy that takes over.
- A large action std (0.6+ on every joint) is not by itself a sign of instability here: 65% of
  deterministic floor actions are saturated at the +-1 limit, so the body needs near-maximum torque
  to rise. The unstable part is the update size (`approx_kl` ~0.4, `clip_fraction` ~0.75), which is
  still unresolved.

**Open.** (1) Train the stand policy from the floor policy's hand-off states. (2) Build the switch
and a learned hand-off boundary ("will the stand policy hold 3 s from this state?"). (3) The
unreached start side. (4) Lower the learning rate for the floor policy. (5) Multi-seed
repetition: all results above are single training runs.

**Later the same day (2026-10-12).** The floor policy `v4` finished its budget at 25.9M cumulative steps and
reaches the hand-off pose in **80/100** fixed-seed starts (deterministic, median 1.8 s); the 7.6M / 10.2M /
19M+ points were 24% / 38% / 72-76%. Handing 158 of its hand-off states to the stand policy for 3 s:
`v37`@99M holds 12 of 158 (7.6%) -> 8/100 end to end; `v37`@120M holds 5 -> 3/100. So the floor policy is no
longer the bottleneck.

The stand policy had meanwhile **regressed**: its 100-seed calm gate went 49-51 (90-99M) -> 45 / 41 / 47 / 38
(102 / 105 / 108 / 112M), while its training-log metrics (std 0.47, `r_head_record`, no NaN) looked healthy and
the hand-off catch rate stayed flat (5-9%) across all checkpoints. It was stopped at 120M. Checkpoint choice for
the restart was made by measurement: `v37`@96M (calm 51, hand-off 12/158), the last point before the decline.

`v38` restarts the stand policy from `v37`@96M with 40% of episode starts drawn from a fixed pool of 1,500
hand-off states (`make_handoff_pool.py`, seeds disjoint from the 150-state evaluation set, half sampled
stochastically), 80% calm among the remaining starts, and all floor-exploration devices (bursts, reset
archive, head-record bonus, correlated noise, entropy bonus) switched off. First 2M steps: `approx_kl` settled
from 0.2-0.4 to 0.03; hand-off catch 9/150 vs 11/150 at the start, calm gate 49 vs 51 -- no effect yet, and
calm is not being lost. The pool is static; a failure-weighted refresh is the planned next step if the
train/eval gap opens.


## 6. Update 2026-10-13..15: stand-ready target, a separate prone policy, staged targets

All numbers: deterministic policies, fixed seeds, "held" = the stand policy `v37`@96M takes over at the moment of
hand-off and the torso never drops below 0.7 m for 3 s (150 steps). Seeds 0..199/399 for evaluation, 1000+ for anything trained on.

**1. Hand-off target moved to the region the stand policy can hold.** Reaching "head >= 0.9 m" (v1-v4) handed over deep,
still-rising squats the stand policy caught 5-9% of the time. The floor policy now has to reach the *stand-ready region*:
foot down, torso >= 1.08 m, both knees straighter than -20 deg, |pitch| < 20 deg, |vz| < 1 m/s, |pitch rate| < 100 deg/s
(`FloorEnv(standready=True)`; the region comes from `analysis/stand_velocity_map.py`). Lying-start reach of that region went
2% -> 32% and plateaued; each condition alone is usually met, all six at once rarely, and in 74% of failures no foot is on the
floor at the highest point (the policy throws the body up).

**2. Things that did not work** (all single runs): a height x speed penalty (the "braking habit", share of steps above
1.0 m with |pitch rate| < 100 deg/s, stayed 53% -> 54%); making the episode longer helped only slightly (+6 reached /
+3..4 held per 100 seeds from 300 to 450 steps); a waist-angle condition (AUC 0.44, wrong direction); a centre-of-mass condition
trained into the policy (see 4). **Discovered mid-way:** `env.py` ends an episode as a KO after the torso stays below
`FALL_HEIGHT` for 250 steps, so a policy still on the floor at 5 s never saw the longer limit; `FloorEnv(down_timeout=False)` switches
it off and all later floor numbers are measured without it.

**3. One start side was never solved - and it is solvable.** The sign of the initial torso pitch splits starts 80/20 (159 vs 41
of 200). `v12` reached the region from 87/159 on the larger side and 1/41 on the other (inferred face-down / prone; this assumes the
body faces +x and was not verified directly; positive waist angle moves the head toward -x, checked). A closed-loop MPC
oracle stands from 5/6 prone starts (88-100% of the time standing, first upright after 0.4-0.6 s), so it is a learning
problem, not the body. A fresh prone-only network (`--prone-frac 1.0`) was run several ways:
- from scratch with the strict target: head rises (torso median 0.93 m at 7.5M) but the region is reached 0-1 times in 100
  for 26M steps, and it regresses afterwards;
- **staged target (`FloorEnv(level=...)`, level 0 loose -> 1 strict, tightened by `LevelCurriculumCallback` whenever the training
  success rate clears 0.35), warm-started from that 7.5M network (`prone4`)**: level 1.0 after ~16M steps, then 40% reach;
  continued at a lower learning rate (`prone8`, 42.5M -> ~100M) it reached **~58% of prone starts, ~22% held** (400 seeds, five
  candidate checkpoints between 77M and 98M are indistinguishable; plateau since ~77M);
- the same staged target from scratch (`prone7`, 34M steps): level never left 0 (training success 5.5% < gate), 0/200 at the strict
  target in every one of 12 evaluations. The staging works only on top of a network that already knows how to get high.

**4. What decides whether the stand policy holds a hand-off state.** For 163 prone hand-offs: the six region variables do not
separate held from not held; the centre of mass relative to the front foot does (AUC 0.26: +2 cm ahead of the front foot when held,
+10 cm when not). Adding "COM within 6 cm of the front foot" to the region lifts the held share of reached states from 38% to
56% on a policy not trained for it, but trained into the policy (8M steps) it did not raise the number held. The waist sits
at its +45 deg limit at the highest point in almost all episodes and is not informative.

**5. Controller structure, measured end to end** (natural start mix 80/20, 20 s, "standing" = torso >= 0.95 m and |pitch| < 30 deg
for the last 25 steps): floor policy -> stand policy, no way back **26%**; with a loop back to the floor policy when the torso
falls 24%; the floor policies alone **4%**. The floor policies often fall after reaching the region and get up again
(55% have the torso above 0.7 m at 3 s, only 1% never dropped below), the stand policy does not recover once it falls, and
looping back adds nothing. Current best system = **floor policy (supine `v12`, prone `prone8`) -> `v37`@96M, about 26%
end to end**; the stand policy catches 51% (supine side) / 36% (prone side) of what it is handed, versus 77-92% from upright starts.

**Lessons (also in the Korean doc, section 7):** pick checkpoints by re-measuring a few top candidates on the same larger seed set
(a 200-seed "best" was noise); always state which "held" criterion is meant; hypotheses about causes (penalty, local optimum)
must be marked as such until measured.

`v38` (stand policy continued from `v37`@96M with 40% loose hand-off starts) finished at 136M with no gain: it catches 8/150 of the loose
hand-off evaluation set vs 11/150 for `v37`@96M.

**Next / open.** Retrain the stand policy **from scratch** on a broader start mix (upright, crouched, and ~8,000 strict-region
hand-off states pooled from `v12` and `prone8`, `scripts/make_strict_pool.py`; the precedent `v21` went 5/100 -> 15/100 ->
30-40% calm gate in 8.7M -> 17.7M -> ~55M steps). `v37`@96M stays as fallback. Still open: the crouched-start gap (no policy
owns it; the stand policy holds 5-44% of crouched calm starts), the switch / selector between the three models
(supine floor, prone floor, stand - not built; start side is observable from the initial pitch sign), and multi-seed repetition.

## 7. Update 2026-10-15..17: retraining the stand policy, hand-over timing, a rule-based controller

Evaluation tools: a 8,000-state pool of strict-region hand-off states (prone `prone8` x2 checkpoints + supine `v12` x2 checkpoints, seeds 1000+/20000+) for training and a disjoint
300-state set (seeds 0+, deterministic) for evaluation. The baseline stand policy `v37`@96M keeps 127/300 (42%) of the evaluation states genuinely upright (torso >= 0.7 m, head >= 0.9 m,
pitch < 40 deg held 20 steps).

**1. Stand policy from scratch: nothing learned (six runs, 13-55M steps).** With the x1.5 body, with/without a 40% pool from step 0, with a performance-gated pool (it never opened: 111 gates read 0.000),
with the v21-era reward (`STANDAI_NO_RECOVERY_GATE=1`), with a torque-normalised std cap (0.4): fixed-seed upright-start gate 0 throughout; the policy ends up flat on the floor (torso 0.40 m after 3 s). The training log looked
healthy (fall_rate 0.4-0.5, episode length 650-900) because the training-time **knee-angle floor** (calm episodes clamp the knee at -50..-80 deg) acts as an invisible support: re-run in a trainer-like vector env, the KO rate is 16/16
with no scaffold, 10/16 with the knee floor (matching the log). Checks that rule out a broken environment: `v21`@82.4M gives 33 / 40 / 39 of 100 at gear x1 / x1.25 / x1.5 in the current code (recorded: 33 at x1), so the
learned skill survives the stronger body; the failure is of learning from scratch at x1.5, not of forgetting. A faithful v21 replica (x1 body + v21-era reward, `v46`) is paused at 4M steps without a verdict.

**2. Continuing `v37` on the hand-off pool does not raise the catch.** `v38` (loose pool, 40%, 136M): 8/150 vs 11/150. `v47` (`v37`@96M + strict pool 40% + failure-weighted replay + std floor 0.45 + entropy bonus):
127/300 after 10M steps (baseline 127), while the pass rate on pool states rose 26 -> 36% - the policy memorises pool states and does not generalise to the held-out set. Upright-start gate stays ~0.5.

**3. Hand-over timing and the trigger do not help either.** Testing "would the stand policy hold 3 s if it took over here?" at 17 moments around the entry gives an oracle of 85% (prone) / 81% (supine) of reached states
catchable at some moment, but the per-moment rate is only 38-56% (selection effect). Supine is flat at 45-56% from entry to +60 steps; prone peaks at 50% five steps before entry (38% at entry) and collapses to <10% afterwards.
Loosening the region (level 1.0 -> 0.6) raises reach (prone 63 -> 79%, supine 44 -> 58%) but the number held stays at ~22% of starts.

**4. Rule-based controller** (`scripts/controller.py`: pick the floor policy by pitch sign, hand over at region entry, fall back to the floor policy when the stand policy falls): **24% of starts end standing at 20 s with fall-back
(supine 37/159, prone 12/41), 18% without**; hand-over happens in 58% of starts. Earlier A/C comparison (24 vs 26%) pointed the other way, so the fall-back effect is within the noise (+-6).

**Where this leaves the project.** Both floor policies get to the stand-ready region in 44-63% of starts; the stand policy then holds about 42% of those, and nothing tried (fine-tuning on the pool in four variants, trigger and timing,
loosened regions, a COM condition in the floor policy, training the stand policy from scratch) moved that number. States it loses have the centre of mass ~10 cm ahead of the front foot (AUC 0.26), i.e. need a step; whether
the stand policy can learn that is open. Open: the paused x1 replica (`v46`), a dedicated capture-step policy, a learned hand-over selector (only worth it if a classifier on the labelled snapshots reaches AUC >= 0.8; not tried).

## 8. Update 2026-10-17: the knee floor is a crutch; removing it lets a stand policy learn from scratch (so far); one-command benchmark

**The knee floor.** During stand training, calm (upright-start) episodes clamp the knee angle at -50 -> -80 deg every step (`KneeScaffoldCallback`; a position clamp, not a force). The body model's real knee range is -140 deg
(a human knee flexes roughly 135-150 deg, not verified against a source), so the clamp keeps only the upper ~57% of the model's flexion. It was added because policies retreated into a deep crouch within a second
(the "escape hatch" comment in `locomotion_env.py`); the working hypothesis (the user's) is that the x1 body was too weak to rise from a deep crouch, so the clamp compensated for it. Evaluation, hand-over, the
controller and the floor environments never apply it, so a policy trained with the clamp leans on it: re-run in a trainer-like vector env the same checkpoint is KO'd in 16/16 episodes with no scaffold and 10/16 with the clamp.

**`v48`** = the failed `v45` setup (x1.5 body, v21-era reward without the recovery gate, std cap 0.4, no pool) with only the clamp removed (`--no-knee-floor`). Six earlier from-scratch runs (with the clamp) stayed at gate 0 up to 55M steps;
`v48` leaves 0 at ~5M steps. Fixed-seed alarm gates (30 upright starts): 0 -> 0.17 (5M) -> 0.30 (7M) -> 0.47 (10M), then flat at 0.37-0.47 up to 16.5M.

**One-command benchmark** (`scripts/analysis/benchmark.py`, ~4 min, 200 seeds): stand catch on the 300 held-out hand-off states, upright-start gate split by knee bend, both floor policies' reach and hold, controller end-to-end.

| | `v37`@96M (reference) | `v48`@16.8M |
|---|---|---|
| hand-off states held upright (of 300) | **127 (42%)** | 88 (29%) |
| upright-start gate (100 seeds) | **51** | 41 |
| by knee bend: >= -10 / -25..-10 / -40..-25 / < -40 deg | **90 / 68 / 43 / 23 %** | 71 / 54 / 40 / 16 % |
| supine / prone floor policy -> stand holds 3 s (share of starts) | **28 / 22 %** | 19 / 12 % |
| rule-based controller, standing at 20 s | **24%** | 20% |

`v48` is below `v37` everywhere, but `v37` has ~96M steps and `v48` 16.8M (the from-scratch precedent `v21` had 15/100 at 17.7M). The 30-seed alarm gate (0.37-0.47) had looked close to `v37`; the benchmark does not agree, so alarm gates are
only a direction check. Next decision point: benchmark `v48` every ~10M steps; if the hand-off catch passes 127 it replaces `v37`, if it plateaus the project is back at the catch wall of section 7. If it works, the stand policy is also the only
owner of crouched starts, which the table shows to be its weak spot (23-43% below -25 deg).

**Compatibility check for retuning the stand policy:** the floor policies share only the body (gear), the 70-dim observation and the action semantics with the stand policy; rewards, exploration (std), pool share and curriculum are
stand-training-only (`FloorEnv` discards the parent reward), so those can be retuned without touching the floor policies. Jerk/effort penalties are not part of the stand-only reward (the proposal to scale them with gear was moot).
New options: `train_locomotion.py --no-knee-floor`, `train_floor.py --supine-frac` (supine-start-only floor training, not run yet), `scripts/controller.py` + `controller_eval.py`.

## 9. Update 2026-10-18: a lower learning rate stabilises the stand policy (20 s standing 29% -> 41%); the 80:20 start split is an artefact

**Instability, not degradation.** The from-scratch stand run `v53` (start distribution matched to the hand-off states: forward/joint velocity noise, 30% of starts blended from two hand-off states, no lying starts) had a hand-off catch that jumped
by ~90 (of 600 held-out states) between checkpoints 1M steps apart (`v37` = 260): 277 at 10M, 291 at 20M, ~215 on average at 25-55M, 150-177 after 56M, while every training reward and the fall rate were flat after 20M.
Cause: PPO at the default learning rate 3e-4 ran at `approx_kl` 0.08-0.1 and `clip_fraction` ~0.5 (healthy: 0.01-0.03 / 0.1-0.2). The stand trainer had no learning-rate option and had never been run with a reduced rate (only the floor policy `prone8` was).
This is standard PPO practice, not a new finding; the project had simply missed it for the stand policy.

**`v54`** = `v53` @10M continued at 5e-5 (`train_locomotion.py --learning-rate`): `approx_kl` 0.012-0.015, checkpoint-to-checkpoint jitter +-10 instead of +-50. Rolling controller, standing after 20 s (200 seeds per start side):

| | supine start | prone start | 50:50 average |
|---|---|---|---|
| `v37` @96M (old stand policy) | 28% | 30% | 29% |
| **`v54` @20.3M (new stand policy)** | 42% | 40% | **41%** |

Same checkpoint family at 17M / 20.7M: 39% / 41%. Hand-off catch (150 states per side): supine 69% (v37 53%), prone 26% (v37 34%). Weaker than v37 on upright-start gate (46 vs 51/100) and prone catch. Baseline file: `checkpoints/best_stand_v54.zip`.

**The 80:20 supine/prone split of the environment's lying starts is an artefact of the collapse generator**, not a modelled fall distribution: the hip target range is flexion-heavy (-20..+120 deg, targets drawn from 0.6x the range), so bodies fold backwards.
Ablation: 80% supine as is, 91% without the root-pitch jitter, 52% with hip target 0, 47% with a symmetric hip target. The pitch-sign label matches the physical orientation in 100% of 1,500 starts. The benchmark now reports each side and a 50:50 average;
the old "controller 24%" (80:20 weighted) is 28% / 30% per side for `v37`. The real fall direction mix is not known.

Open: supine floor policy `ppo_floor_supine2` (from scratch, supine starts only, staged target, lr 1e-4) is training; floor reach (56% supine / 63% prone) has not moved in many experiments and is now the larger loss at 50:50 (40.5% of starts never reach the hand-off region vs 34.5% reach but are not held).

## 10. Update 2026-10-19: supine hand-offs raise a leg, the stand policy is left-right asymmetric (fixed without retraining), a critic-based aux reward did not help

- **Supine floor policy (`v12`) hands over standing on one leg with the other raised to hip height in 100% of its hand-offs** (the region only needs one foot down); prone hand-offs are 95% two-footed. The stand policy `v54` catches 73% of v12's hand-offs (left leg raised) but 8% of their left-right swapped copies, and 4% of a re-trained supine policy's (right leg raised) hand-offs.
- **Symmetrisation (`scripts/sym.py`)**: left and right limbs are identical, so swapping them gives a valid state. The wrapper acts on whichever of (observation, limb-swapped observation) the stand policy's own critic values more and swaps the action back: 109/150 on both v12's hand-offs and their mirrored copies (raw policy 109 / 14).
- **Why the stand policy misses prone hand-offs (26-36% catch)**: the centre of mass is ahead of the front foot (catch rate 84% below -5 cm, 46% for -5..0 cm, 22% for 0..5 cm, <=10% above, 0% above +15 cm). Waiting for the COM to come back before handing over does not help (a controller gate lowers standing 53% -> 45-52%).
- **The stand policy's critic predicts what it catches** (AUC 0.95 prone, 0.90 supine hand-offs). Used as an auxiliary reward for the floor policy (hand-off bonus 10 x V_norm + a head-height-weighted per-step term, symmetric value), it did not move the arrival: re-trained supine policy `b3` reaches the strict region in 40% (v12 56%), is caught in 28% (v12 66-69%), controller supine standing 10% (v12 46%). Best system unchanged: v12 + prone8 + v54, hand-over level 0.7, ~50% standing after 20 s (50:50 supine/prone).
- Practical notes: a fresh floor policy must learn to rise at the default learning rate (1e-4 / 3e-5 never left level 0); a speed penalty of 0.08 stalls the staged target at ~0.6 (removing it unblocked 0.6 -> 1.0); a two-feet condition tightened through the level curriculum from the first stage costs nothing in stage 1 (level 1.0 after 3.8M steps, 44% reach) but the resulting hand-offs (loose stage-1 target) are not catchable.

## 11. Update 2026-10-19: a COM-ahead ladder + left/right mirroring + a step shaping reward on the stand side: no regression, small gain (controller 20 s standing 50% -> 54%), the far-ahead hand-offs stay at 0%

Design principle: the floor policy only delivers, balance (including stepping) is the stand policy's job, so the prone hand-off problem (catch 23-30%, COM ahead of the front foot) was attacked on the stand side (`ComLadderPoolCallback`, `STANDAI_STEP_K`; docs 3.23).
- **Ladder + mirror only (v55a)**: the pool's prone half sorted by COM-minus-front-foot, offered easy to hard, plus left/right swapped copies. The ladder stalled at +0.05 m (boundary pass 0.15-0.26); prone catch 23% (v54 26%), calm gate 45, controller 50%. No regression, no gain.
- **Stepping already happens**: in missed prone hand-offs 82-99% of stochastic rollouts move the trailing foot past the initial front foot (median foot travel 0.25-0.43 m), so exploration is not the missing piece; the metric cannot tell a controlled step from being dragged by a fall.
- **+ step reward (v55b, `STANDAI_STEP_K=300`: potential on |COM x - mean foot x|, paid only with torso >= 0.7 m and |pitch| < 40 deg)**: prone catch 27%, supine 72%, all hand-offs 148/300 (v54 144), calm gate 48, controller 20 s standing 54% (prone 59%, supine 50%). On the independent set eval2_all: prone 29% (v54 26%, v55a 21%), supine 74% (73%); by COM-ahead bin only +0.05..0.10 m moved (5% -> 26%, 1 -> 5 of 19 states); **everything above +0.10 m (43% of prone hand-offs) is still 0%**.
- Within the noise of 150 states per side (+-3-4 points) the gain is small. Candidates: continue, restart from scratch with all ingredients, or a separate hand-over policy for the cases that need several steps. Checkpoints: `best_stand_v55a.zip`, `best_stand_v55b.zip`; tools `analysis/step_diag.py`, `analysis/catch_by_com.py`.

## 12. Update 2026-10-19: a separate hand-over policy between the floor policy and the stand policy: prone hand-off catch 29% -> 87%, prone-side standing at 20 s ~49% -> 70-74%, 50:50 average ~50% -> ~58-61%

The stand-side attempts (ladder, mirror, step reward; section 11) stayed within noise, so the delivery was split into a third policy: **floor policy -> hand-over policy -> stand policy**. The hand-over policy (`scripts/handover_env.py`, `train_handover.py`) starts from pooled hand-off states and has to reach a state the stand policy holds (COM inside the support interval of the grounded feet, torso >= 1.0 m, |pitch| < 25 deg, slow, held 10 steps); observation and action are the stand policy's.
- **Success condition matters**: "COM behind the front foot" failed for supine hand-offs (one leg raised, the raised foot counted as the front foot: catch 27% when satisfied); with grounded feet only and the COM inside [rear - 8 cm, front - 3 cm] the condition predicts the stand policy's catch on the prone pool (78% vs 28%).
- **From scratch it does not learn** (two runs, 0% success; one was a reward-design mistake: a per-step torso reward summing to more than the success bonus rewards surviving instead of succeeding); the body collapses forward into a stable 0.65 m crouch. **Warm-started from the stand policy v55b** it succeeds on 59% of the easiest pool states at once and the COM-ahead ladder runs through the whole prone pool (up to +0.49 m) within 6M steps.
- **Result (prone, independent eval set, deterministic)**: stand policy alone catches 29% of the prone hand-offs, hand-over -> stand 87% (hand-over success 88%, the stand policy then holds 99-100%). Above +0.10 m COM-ahead the stand policy alone caught 0%; the chain caught 44% at +0.10..0.15 m in the middle of training, more later.
- **Whole system (floor -> hand-over -> stand, hand-over only after the prone floor policy, 20 s, level 0.7, 100 seeds per side)**: fresh seeds 9000+: supine 48% (no hand-over policy), prone 74%, 50:50 **61%**; seeds 7000+: supine 44%, prone 70-71%, 57.5% (without the hand-over policy: ~46-50%). Putting the prone-trained hand-over policy after the supine floor policy lowers it (44 -> 40%).
- **A supine hand-over policy did not help** (63M steps): chain catch 74% -> 77%, 20 s standing 44 -> 45%; the stand policy's falls per start dropped 0.66 -> 0.10 but the hand-over policy collapses instead (35%). The supine side is limited by the floor policy: 27-30% of starts never reach the hand-over region, the median arrival is 5-7 s.
- Funnel tool: `scripts/analysis/funnel.py` (reached / hand-over ok / stood at some time, 10 s, 20 s / falls / end state). Hand-over policy checkpoint: `checkpoints/best_handover_prone.zip` (h3 @66.55M); `controller.GetUpController(..., handover_ckpt=...)`, `benchmark.py --handover`.
- **Addendum 2026-10-20 (supine side)**: a supine hand-over policy trained on LOOSE-level arrivals (level 0.3, 10,000 v12 arrivals; `checkpoints/handover_supine_s2.zip`) lifts the loose levels (level 0.3: 34% -> 48% standing at 20 s, seeds 9000+) but ends at the same ceiling as level 0.7 without it (48%). Diagnosis: 50% of its episodes collapse BACKWARD within ~0.9 s; arrivals with torso pitch > +10 deg (a third of the pool) succeed 2-7%, those leaning forward 68-77%; COM > 0.25 m behind the front foot 0-5%. The floor policy hands over unrecoverable (back-leaning) states. The hand-over critic V predicts success (AUC 0.884 vs 0.778 pitch, 0.855 COM-ahead), but gating the hand-over on V (`GetUpController(..., v_gate=...)`) leaves the supine result unchanged (46-49% at every gate and level, seeds 11000+). The supine side is limited by the floor policy, not by the hand-over or stand policies.
- **Final measurement 2026-10-20 (untouched seeds 13000+, 200 per side, 20 s, level 0.7)**: stand v54 without hand-over policy: supine 47% / prone 48.5% (50:50 47.8%); stand v55b without: 49% / 60.5% (54.8%); v55b + prone hand-over policy (`best_handover_prone.zip`): 49% / **76%** (**62.5%**). The +14.7 points split into +7 from the stand policy v54 -> v55b (prone +12 points, ~2.4 sigma at 200 seeds: the gain called within noise in section 11 is real) and +7.7 from the hand-over policy. Funnel (200 each): supine reaches the hand-over region 74%, stands at 20 s 49% (26% never reach, 18% reach but end down); prone reaches 94%, hand-over success 68% / collapse 26%, stands 76% (6% never reach, 8% reach but end down, 10% stand once and end down). Reproduce: `STANDAI_NO_RECOVERY_GATE=1 .venv/bin/python -I scripts/analysis/funnel.py --ho-prone checkpoints/best_handover_prone.zip --n 200 --seed0 13000`.
- **Addendum 2 2026-10-20**: a supine floor fine-tune with the hand-over policy's critic as an auxiliary reward (sv1: v12 + `STANDAI_STAND_VALUE_*` pointing at `handover_supine_s2.zip`, 20M steps, stable) did not move the supine result: standing at 20 s (seeds 15000+, 200): v12 43.5% / v12+s2 45% / sv1 39.5% / sv1+s2 47.5% (within noise; sv1+s2 is faster at 10 s, 67% vs 51%). Diagnostic for "reward the upward force": predictive power for hand-over success (AUC): COM vertical velocity 0.657, feet load/weight 0.431, critic V 0.873. Every supine-side intervention (hand-over level, hand-over policies s1/s2, critic gate, sv1) ends at 43-50%.
- **Addendum 3 2026-10-21 (why the supine floor policy is capped at 43-50%)**: 25-30% of supine starts never reach the hand-over region in 20 s and the hand-over policy cannot rescue them (0%); the outcome is sensitive to tiny perturbations (action noise 0.1: reach 69% -> 57%; 85% of starts reach only sometimes; mirroring the start flips the outcome for 37% of starts, either-frame success 87% vs 68%), but retrying with noise/sampling saturates at 74-75% and flipping the left-right frame every 5 s helps the floor policy alone only slightly (+6-10 points) and hurts end-to-end (45% -> 30%, interaction with the pitch-sign policy switch). Start posture (flat legs 64% vs bent 72%), arm use (arms essential: zeroing them 66 -> 2/100 reached, but pushing force only 0.14 x weight, not different between successes and failures; shoulder range -50..160 deg is human-like and reaches the floor in 70% of supine starts at the back limit), trunk unloading, total ground reaction force (peak, mean) do not separate v12's supine successes from failures (AUC 0.47-0.62) while they do for the prone policy (0.87-0.96): the prone policy succeeds by pushing up hard with the arms (5x weight peak, arm load 0.60 x weight), the supine one by timing. A from-scratch probe with an explicit arm-push reward (ap1) is in progress. Tools: `scripts/watch_failures.py`.
- **Addendum 4 2026-10-21**: ap1 (from scratch with an arm-push reward, 8M steps) did change the strategy (arm load 2.3x that of s4a at 5.5M) but its loose reach target does not match the controller's stand-ready region (1-3/100 arrivals); judged fairly on its own target it reaches 51% (v12 99%, s4a 85%) and stands 0%. The union of 8 v12-lineage checkpoints over the same 100 starts is 90% (single 52-64%) but rotating checkpoints on a stall gives 69-71% (single 71%): a hindsight bound, not an achievable gain. With the official controller (seeds 23000+, 200): loose-target hand-over 40.5% / with s2 47% / s2 keeping control 39%; level 0.7: 47% / 50.5% / 49%; prone with the hand-over policy keeping control 78.5%. An ad-hoc evaluation loop gave 66% for the same configuration and was an artefact (check every new measurement loop against the official controller). Floor friction (coefficient 1.0 everywhere) is not the bottleneck (supine reach 51/56/57/51/23 at 0.4/0.7/1.0/1.5/2.5). The supine side stays at ~45-50% for every intervention tried.
