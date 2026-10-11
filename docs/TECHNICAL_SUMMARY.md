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
