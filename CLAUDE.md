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
**State of play (2026-10-15):** three policies (details: `README.md` status table, `docs/experiments_2026-10-11.md` 3.17, `docs/TECHNICAL_SUMMARY.md` 6).
- Supine floor policy `ppo_floor_v12` (best used: `autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip`): reaches the stand-ready region from 55% of its starts.
- Prone floor policy `ppo_floor_prone8` (`checkpoints/best_prone8.zip` = 86.5M; 77.45M / 82M / 84.25M / 97.8M are statistically the same, ~58% reach, ~22% held): training stopped at ~100M, plateau since ~77M.
- Stand policy: `v37`@96M is the one used for hand-over measurements. `v38` (hand-off pool 40%) FINISHED at 136M with no gain: it catches 8/150 of the loose hand-off set vs 11/150 for `v37`@96M, calm gate (30 seeds) 0.37-0.53. A from-scratch retrain on a broader mix is next (pool files `checkpoints/pool_*.npz` from `scripts/make_strict_pool.py`).
- Best end-to-end system today: floor policy by start-pitch sign -> `v37` (~26% end up standing; a loop back to the floor policy on a fall gives nothing, floor policies alone 4%). The selector/switch is not built.
- Floor runs use `--no-down-timeout` (env.py's 5 s "stayed down" KO off), `--max-steps 450`, `--standready`; `--prone-frac 1.0` = prone starts only; the staged target is `--level-start 0.0` (works only from a network that already gets high).
- Pick floor checkpoints by re-measuring candidates on the SAME 400 seeds (`scripts/analysis/eval_ckpt.py`); 200-seed "bests" were noise. State the "held" criterion every time (never below 0.7 m for 3 s vs torso >= 0.7 m at 3 s differ hugely for the floor policies).
Check for running processes first (above). As of this edit only the strict-pool builders (`make_strict_pool.py`) may be running.
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
