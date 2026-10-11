"""Single-fighter mobility/recovery environment.

'a' trains to (1) regain a stable standing state from an arbitrary starting body
configuration (anywhere from a light shove to fully collapsed on the ground) and (2)
track a commanded x-velocity while doing it, with 'b' teleported off to the side and held
at zero control -- physically present in the model (it always has two bodies) but never
interacted with.

Validates a hypothesis raised after the 10.29-10.32 fall investigation: the persistent
"clean fall" plateau (85% of falls, no preceding stagger, no clear single cause -- docs
10.31) might be a single combat policy splitting its capacity between locomotion/mobility
and combat objectives, rather than a locomotion skill that just hasn't been trained
enough. Training balance/recovery/mobility in total isolation, with a much broader
curriculum than the existing get-up curriculum (GETUP_CURRICULUM_PROB, docs 10.23)
covers, is a cheap way to check: if a solo policy is dramatically more fall-resistant
than the combat-trained one at a comparable step budget, that's real evidence for a
hierarchical (locomotion + combat) architecture instead of one shared policy learning
everything at once.

v2 (docs 10.33): added velocity-command tracking -- v1 only rewarded NOT falling, with no
signal at all for actually moving, so it had no reason to learn real mobility (the
`root_x` actuator was available but unrewarded). Bumps the observation space by 1 dim
(the current target velocity), so this is NOT warm-start-compatible with v1 checkpoints.
"""
import mujoco
import numpy as np
from gymnasium import spaces

import env as env_module
from env import FALL_HEIGHT, Fighter2DEnv, GETUP_SETTLE_STEPS

# Combat reward terms are meaningless with 'b' parked far away and inert. Zeroing these at
# module level is per-process only (SubprocVecEnv gives each worker its own copy of the
# module), so this doesn't affect env.py's other users. Left un-zeroed, ENGAGE_PENALTY_SCALE
# in particular would apply a large constant penalty (foot_dist stays huge once 'b' is
# parked) that has nothing to do with the actual task.
env_module.ENGAGE_PENALTY_SCALE = 0.0
env_module.PROGRESS_REWARD_SCALE = 0.0
env_module.DAMAGE_REWARD_SCALE = 0.0

# v2 fair-eval (docs 10.34) found the trained policy crouches at ~knee limit (-140deg)
# essentially always, even starting from a normal standing pose -- not just during actual
# recovery. RECOVERY_REWARD only engages below FALL_HEIGHT (a deep knee bend alone doesn't
# drop the torso that far, so it never fires here), and velocity-tracking reward is
# height-agnostic (crouched or standing, matching the target speed pays the same), so there
# was very little pull back to full standing once a knee bend became "good enough" to be
# stable. Doubling both height and knee-avoid pressure specifically for this task -- pure
# reward-constant changes, so this continues from the existing checkpoint instead of
# retraining from scratch.
env_module.HEAD_HEIGHT_REWARD_SCALE = 6.0   # was 3.0
env_module.KNEE_AVOID_SCALE = 1.0           # was 0.25, then 0.5 (docs 10.34) -- still measured
                                             # knee_r straighter than -90deg only 6% of the time
                                             # and mean |actual vx| only 0.158 m/s despite targets
                                             # up to 1.0 m/s (docs 10.35), so the permanent crouch
                                             # is winning even at 2x. Pushing further.

B_PARK_X = 6.0  # far enough that a/b geoms can never contact regardless of b's pose

# ---- calm/harsh episode split (docs 10.36/10.5x) ----
# v13curriculum changed BOTH CALM_EPISODE_FRAC (0.6->0.45) and HARSH_SEVERITY_LO (0->0.5) at
# once: deep-collapse a_out improved (67.5-72.5% -> 57.5-67.5%) but calm a_out got WORSE
# (20% -> 40%). v13b isolated the severity-floor half alone (CALM_EPISODE_FRAC reverted to
# 0.6, only HARSH_SEVERITY_LO=0.5 kept): calm a_out went right back to 20%, but deep-collapse
# a_out did NOT improve (70-75%, same as baseline within noise) -- the floor change contributed
# nothing on its own. So the deep-collapse gain in v13curriculum came from the OTHER half
# (more harsh-episode exposure via a lower CALM_EPISODE_FRAC), which is the real calm-vs-harsh
# tradeoff, not a free win. This run isolates that other half: CALM_EPISODE_FRAC lowered,
# severity floor reverted to 0 (original uniform [0, severity_hi]).
CALM_EPISODE_FRAC = 0.45       # was 0.6 -- see above; the half of v13curriculum not yet
                                # tested alone
CALM_SEVERITY_HI = 0.3         # calm episodes still get SOME light random perturbation
                                # (severity sampled in [0, this]), just nothing collapse-grade
HARSH_SEVERITY_LO = 0.0        # reverted -- measured to contribute nothing on its own (see
                                # above), original uniform [0, severity_hi] draw

# ---- velocity command ----
TARGET_VX_RANGE = 1.0          # m/s, sampled uniform in [-range, +range] (0 = "hold position"
                                # is just as likely to be sampled as any other value, not a
                                # separate special case)
RESAMPLE_EVERY_STEPS = None    # was 200 (docs 10.41): re-rolling mid-episode meant no single
                                # command was ever committed to for long, so "twitch briefly
                                # toward whatever's asked" could already capture most of the
                                # achievable reward -- None now means only reset() rolls a new
                                # target, so a whole episode (up to MAX_STEPS=1000) means
                                # actually committing to one sustained walk, not several stabs.
VELOCITY_REWARD_SCALE = 5.0    # was 2.0 (docs 10.35) -- measured mean |actual vx| only 0.158
                                # m/s despite targets sampled up to 1.0 m/s, i.e. the policy
                                # barely tries to move at all. 2.0 wasn't enough to outweigh
                                # the safety of standing (crouched) still; pushing to 5.0 so
                                # actually matching a fast command is worth clearly more than
                                # the other per-step rewards combined.
VELOCITY_SIGMA = 0.3           # was 0.5, then 0.7 (docs 10.35) -- the widen-it attempt was
                                # backwards: even after the knee/crouch problem got fixed
                                # (v7, docs 10.40), mean |actual vx| was STILL only 0.129 m/s,
                                # unmoved by any of the crouch fixes -- a separate problem.
                                # The wide sigma was giving real "free" credit for NOT moving:
                                # against a 1.0 m/s target, standing still (vx=0) scored
                                # exp(-1.0^2/(2*0.7^2)) = 36% of max reward doing nothing.
                                # Narrowing to 0.3 drops that to ~4%, so standing still against
                                # a real command stops being a viable safe default.
                                # gaussian shape (always >= 0, same cliff-free family as the
                                # existing height/stability rewards) rather than a raw squared
                                # penalty, for the same unbounded-blowup reasons documented at
                                # env.py's height-penalty history (docs, "Direct height PENALTY")

# ---- gait shaping (docs 10.46/10.47/10.48) ----
# The outcome reward (velocity, above) never got the policy to actually try leg-stepping --
# mean |actual vx| stuck at ~0.13-0.17 m/s across seven straight reward/curriculum/assist
# changes (v2 through v9). Tried shaping directly on the PROCESS instead of the outcome:
# -hip_r_vel*hip_l_vel (positive when the two hips rotate in OPPOSITE directions, clamped at 0
# otherwise), meant to reward the core signature of a stepping gait without being satisfiable
# by e.g. vibrating both legs in sync. Wasn't strict enough on its own -- v10 (10.47) measured
# r_gait climbing hard (130->340 over 8M steps) while mean |actual vx| and its correlation
# with target_vx both went DOWN, and fall_rate crept up too: marching/scissoring in place
# (real anti-phase hip motion, zero net displacement) satisfied this term just as well as
# actual walking. v11 (10.48) re-gates it by multiplying with velocity_credit (below) -- can
# now only pay out while ALSO tracking target_vx reasonably well, closing the loophole
# directly instead of hoping a bigger/stricter version of the same ungated shape works.
#
# v12fresh (docs 10.50): the raw product -hip_r_vel*hip_l_vel is UNBOUNDED -- measured up to
# 83 mid-training (p90=37, p99=64), versus every other reward term in this env being
# explicitly capped (height's ratio, stability's max(0,1-x)^2, velocity's gaussian). fall_rate
# hit 0.96 and r_gait exploded to 5000+/episode: the policy found that flailing both legs as
# violently as possible in opposite directions paid far more than anything else, fall risk be
# damned. Saturating it the same cliff-free way the rest of the reward function already does
# (1-exp(-x/ref), same family as height/stability/knee_avoid) removes the incentive to
# escalate past a reasonable pace -- max payout is now the same regardless of whether the hips
# move at a brisk-walk speed or a flailing one.
GAIT_REWARD_SCALE = 3.0
GAIT_REF = 2.0                 # raw -(hip_r_vel*hip_l_vel) at which the saturating curve is
                                # ~63% of its ceiling -- picked well below the flailing values
                                # (p90=37) that caused the runaway, in the plausible range for
                                # a deliberate, controlled step rather than a wild swing

# ---- fall brace reflex (docs 10.5x) ----
# While the torso is dropping fast, reward straightening the knees -- the policy otherwise
# buckles like a folding stick when it falls (hips and knees both bend further as it goes
# down, measured on the stand-only checkpoint). Gated on downward torso speed so it only pays
# out during an actual fall, and bounded (0..BRACE_REWARD_SCALE per step) like the other terms.
BRACE_REWARD_SCALE = 1.0
BRACE_VZ_REF = 1.0             # m/s downward torso speed at which the fall gate saturates

# ---- knee straightness, ungated (stand-only, 2026-10-09) ----
# brace above only pays for straightening the knees WHILE actively falling fast -- measured on
# a v17 checkpoint (phase 1, calm-only curriculum) that it had instead found a stable LOW
# equilibrium (torso_z ~0.25-0.36m, well under FALL_HEIGHT=0.70) by keeping both knees pinned
# near their -140deg flexion limit with near-zero velocity -- brace's fall_gate is ~0 there
# (it's not falling, it's just staying down), so nothing was pushing it to straighten back up.
# This term pays for knee straightness directly, regardless of velocity or current height, so
# that equilibrium stops being free -- reuses the same knee_straight ratio brace already
# computes, just without the fall_gate multiplier.
#
# TODO (flagged 2026-10-09, before building the walking curriculum): this is a stand-only-
# phase-0 scaffold, not a permanent preference -- real walking needs the knee to cycle through
# flexion every step, and an always-on reward for straightness would fight that normal gait
# motion once velocity/gait rewards come back. It should be annealed down toward 0 (same
# LinearScaffoldCallback machinery as the knee floor / velocity assist) specifically before
# reward weights are set up for walking -- don't forget.
#
# Raised 1.0->3.0 (2026-10-09): doubling-then-some the height reward (6.0->10.0->~13 so far,
# still ramping toward 15) had ZERO measurable effect on knee angle -- direct measurement on
# several checkpoints along that ramp still showed both knees pinned at -140/-141deg every
# time, even as max_stand_streak_steps started climbing (from something other than leg
# extension, apparently). Reading: height reward doesn't reach the knee-bend habit at all, most
# likely because the policy isn't exploring leg extension in the first place, so a bigger prize
# at a destination it never visits doesn't help. knee_straight directly rewards the mechanism
# itself rather than the end state, and had never been raised since its original value --
# matching it to height/stand_streak's scale instead.
KNEE_STRAIGHT_SCALE = 3.0

# ---- torso upright (stand-only, 2026-10-09; gated by recovery_gate as of 2026-10-10) ----
# root_ry (torso pitch) has no RL-controlled actuator of its own -- in combat it's held by an
# always-on external PD (env.py BALANCE_KP/KD), which stand_only turns off entirely (docs
# 10.5x: it's not a force a real body has). A real human doesn't have one either, though --
# torso orientation is controlled indirectly, via hip/ankle/knee torque creating the right
# ground-reaction-force moment, the same joints already in this action space (plus `waist`).
# So the actuation to do this exists; watching a trained checkpoint directly (stand_only=True,
# matching knee floor, no viewer bug) showed the torso tumbling/rotating with no resistance --
# consistent with the policy not yet having learned that indirect control, not a missing
# actuator. A prior reward keyed on |root_ry angular velocity| was tried and dropped in COMBAT
# (docs: falls there were vertical-collapse-dominated, so it was "measuring the wrong axis") --
# but that was with the balance assist likely still active and a different failure mode; here,
# with no assist and rotation itself visibly the problem, the same axis is probably the right
# one to reward now. Angle-based (not velocity) for the first pass, same cliff-free squared-
# ratio shape as height/stability.
TORSO_UPRIGHT_SCALE = 2.0
TORSO_ANGLE_REF = 0.5          # rad (~29deg) -- ratio saturates to 0 beyond this tilt

# ---- centre-of-mass over support (stand-only, docs 10.5x) ----
# Reward for keeping the whole body's centre of mass horizontally over the feet (bounded, cliff-free).
# Only the horizontal position counts, so it doesn't reward crouching the way a height term would.
COM_REWARD_SCALE = 1.0
COM_MARGIN_REF = 0.10          # m inside the support edge at which the reward saturates

# ---- continuous stand streak (stand-only, 2026-10-08) ----
# height/stability/com/brace all pay out per-step regardless of history, so a policy that
# falls and gets back up every ~1-2s scores almost as well as one that never falls at all --
# measured directly on v7 at 50M steps: training-log fall_rate down to ~0.1 and ep_len_mean
# up near MAX_STEPS, but the real fixed-seed/deterministic 3-second-survival gate was still
# 0% (median fall at step 28, same as the very first stand-only checkpoints). DOWN_RECOVERY_STEPS
# gives it 5 whole seconds to recover before a_out ends the episode, which is plenty of room to
# cycle fall->recover->fall indefinitely without ever being penalized for breaking a stand. This
# term pays directly for the thing the gate actually measures: steps spent continuously above
# FALL_HEIGHT (raised 0.55->0.70 in env.py the same day, for exactly this reason -- see its
# comment there), ramping up to the full scale once that streak reaches the gate's own
# 3-second (150-step) window, and dropping back to 0 the instant the streak breaks -- so every
# fall now costs whatever streak had been built up, not just a brief dip in height/stability.
# ---- ground shaping (stand-only, opt-in via ground_shaping=True, 2026-10-11) ----
# Measured per-step reward by torso height on harsh starts (v23, 111.5M steps): lying flat
# (z<0.2) paid 2.58/step vs 1.96 at 0.2-0.35 -- a valley, so a partial rise cost reward. The
# gap was mostly com_reward (0.82 flat vs 0.28 -- a support-polygon term that means nothing
# lying down), and height's squared head ratio has almost no slope when low (~2.2/m at head
# z=0.3 vs ~7/m standing; 0.11->0.33 across the whole 0-0.5m torso range). So: gate com_reward
# by recovery_gate like knee_straight/torso_upright, and add a linear low-height term that
# saturates at GROUND_HEIGHT_REF (flat above it, so standing-range rewards are untouched).
# knee_avoid is deliberately NOT gated -- it pays for knee height, which is the tucked-leg
# posture getting up actually needs.
GROUND_HEIGHT_SCALE = 3.0
GROUND_HEIGHT_REF = 0.8        # head z (m) at which the linear term saturates

# ---- getup reward (stand-only, opt-in via getup_reward=True, 2026-10-11) ----
# Mirrors the cost that made a receding-horizon planner (predictive sampling, exact physics)
# actually stand up from flat starts (2/6 solved; open-loop CEM with torso-z or cos(pitch)
# objectives never did -- it found headstands, mid-air leaps, or stalled at sitting height):
#  - head-height credit only while a foot is on the ground (no airborne / head-supported credit)
#  - weight-shift term: while low and foot-supported, reward COM near the feet (sit-to-stand
#    needs the COM over the feet before pushing up)
#  - penalties for the head touching the ground and for an inverted torso
#  - stand_streak only counts UPRIGHT standing (pitch + head height), not just torso z
#  - no pitch credit while low -- the forward lean mid-rise is not penalised
# ---- body spec (2026-10-11) ----
# Limb actuator gears are multiplied by BODY_GEAR_SCALE for every LocomotionEnv. Measured: with
# the original gears (x1) a closed-loop MPC oracle stands from only 2/6 lying starts; at x1.5 it
# stands 5/6, and an unchanged policy gains ~+10 points overall on calm starts without losing the
# upright ones (x2+ breaks that policy). So x1.5 is the default body from v28 on. All results in
# docs/experiments_2026-10-11.md before v28 are at x1: set STANDAI_BODY_GEAR=1 to reproduce them.
import os as _os
BODY_GEAR_SCALE = float(_os.environ.get("STANDAI_BODY_GEAR", "1.5"))

# ---- knee gate (stand-only, opt-in via knee_gate=True, 2026-10-11) ----
# Observed on v28: while lying (torso z<0.5) the policy raises its legs high (lowest-knee height peaks
# at 0.58 m, hip swing 2.7 rad/s) and tumbles ~1.4 turns per episode instead of getting up, and
# knee_avoid (reward for knee height) pays ~0.5/step for it. So knee_avoid is faded out by HEAD height:
# 0 at/below KNEE_GATE_HEAD_LO (about a seated head), full at/above KNEE_GATE_HEAD_HI (the head height
# the upright-stand criterion uses). Calm starts keep the head above HI, so they are unaffected.
KNEE_GATE_HEAD_LO = 0.5
KNEE_GATE_HEAD_HI = 0.9

# ---- head-height progress (stand-only, opt-in via head_progress=K, 2026-10-11) ----
# Reward for RISING, not for being high: bonus = K * (gamma * phi(head') - phi(head)) with phi the head
# height capped at HEAD_PROGRESS_CAP (so nothing is paid above it) and normalised to [0, 1]. This is
# potential-based shaping: rising pays, falling costs the same, so rolling/oscillating up and down nets ~0
# and sitting at any height earns nothing extra (the 'sit and farm' optimum a state-based height bonus
# creates). Gives a steep gradient exactly where the per-step reward was flat near the floor.
HEAD_PROGRESS_CAP = 0.9
HEAD_PROGRESS_GAMMA = 0.99     # = PPO gamma; keeps the shaping potential-based

# ---- head-height record bonus (stand-only, opt-in via head_record=K, 2026-10-11) ----
# The potential-based head_progress term nets to zero over a rise-and-fall, so it barely incentivises the
# floor attempts that rarely hold. This one is NOT cancelled: it pays K * (new best normalised head height -
# previous best) only when the head exceeds its best height so far IN THIS EPISODE, and only while a foot is on
# the floor (so leaping/leg-flinging doesn't count). Falling back costs nothing, but a height is paid once, so
# the episode total is bounded by K * (1 - start height) and cannot be farmed by bobbing. Calm starts begin
# with the head already at the cap, so they never pay.
HEAD_RECORD_CAP = 0.9

# ---- burst archive (stand-only, opt-in via set_archive_collect, 2026-10-11) ----
# States reached from a floor start with the head in [LO, HI) m, a foot down and the torso not inverted are
# reported in info["archive_state"] (qpos, qvel, head_z), at most once per ARCHIVE_COOLDOWN steps. The trainer
# keeps them and later starts a share of episodes from them (set_replay_pool), so the policy practises the
# part AFTER the rise from poses the body demonstrably reaches from the floor.
ARCHIVE_HEAD_LO = 0.25        # lowest rung: lying with the head just off the floor
ARCHIVE_HEAD_HI = 1.10        # top rung: 'just stood up' (head past 0.9 m) -- the link that breaks (16 of 19 fall back)
ARCHIVE_GAIN = 0.08           # frontier rule: only states >= this much above where THIS episode started are stored
ARCHIVE_COOLDOWN = 10

REPLAY_JITTER_RAD = 0.03       # joint jitter applied to replayed start states (failed-start replay)

GETUP_SHIFT_SCALE = 1.0
GETUP_SHIFT_REF = 0.5          # m of COM-to-feet horizontal offset at which the term reaches 0
GETUP_SHIFT_MAX_Z = 0.55       # torso z below which the weight-shift term applies
GETUP_HEAD_CONTACT_PEN = 1.0
GETUP_INVERT_PEN = 1.0
GETUP_INVERT_DEG = 120.0
GETUP_UPRIGHT_DEG = 40.0       # stand_streak requires |torso pitch| below this ...
GETUP_UPRIGHT_HEAD_Z = 0.9     # ... and head at least this high
_FOOT_BODIES = ('foot_l', 'foot_r', 'toe_l', 'toe_r')

STAND_STREAK_SCALE = 3.0       # matches HEAD_HEIGHT_REWARD_SCALE's max -- meant to dominate
STAND_STREAK_REF = 150         # steps (3s) -- same window the eval gate uses


class LocomotionEnv(Fighter2DEnv):
    def __init__(self, render_mode=None, stand_only=False, ground_shaping=False, getup_reward=False,
                 knee_gate=False, head_progress=0.0, head_record=0.0):
        self._stand_only = stand_only
        self._knee_gate = knee_gate and stand_only
        self._head_progress = float(head_progress) if stand_only else 0.0
        self._prev_phi = 0.0
        self._head_record = float(head_record) if stand_only else 0.0
        self._head_best = 0.0
        self._start_head = 0.0
        self._collect_archive = False
        self._archive_cool = 0
        # getup_reward implies ground_shaping and adds the foot-contact / weight-shift / exploit
        # terms below
        self._getup = getup_reward and stand_only
        self._ground_shaping = (ground_shaping or getup_reward) and stand_only
        if stand_only:
            # standing has to be learned from the legs and arms alone -- the always-on torso PD
            # assist (env.py BALANCE_KP/KD) would otherwise carry the balance (docs 10.5x: with it
            # removed, the stand-only policy never kept its footing in an episode)
            env_module.BALANCE_KP = 0.0
            env_module.BALANCE_KD = 0.0
            # 0.15 (env.py default) was calibrated against combat's strike reward, not against
            # stand-only's height/stand_streak (both scale 3.0) -- too weak to meaningfully
            # discourage the "tap-dancing" vertical bobbing seen in stand-only training (2026-10-08).
            env_module.STABILITY_REWARD_SCALE = 1.0
            # Raised again, 6.0->10.0 (2026-10-09): directly measured (qfrc_bias at a hand-set
            # upright pose) that holding full leg extension against gravity needs only ~1-5% of
            # each joint's max torque -- extending is nowhere near physically costly. Yet v19
            # checkpoints kept settling into a stable but LOW equilibrium (torso_z 0.10-0.63,
            # std~0 -- genuinely stable, not falling) even with the knee_straight term added.
            # Not a capability problem, a preference one: the current reward magnitude isn't
            # making "stand tall" clearly better than "sit low and stable" in the policy's own
            # value estimate. This raises the stakes on being tall specifically.
            #
            # NOTE: almost set this to 5.0, which would have been a REGRESSION -- the module-
            # level override a few lines up (env_module.HEAD_HEIGHT_REWARD_SCALE = 6.0, from an
            # earlier locomotion-mode experiment, docs 10.33-10.35) already runs at import time
            # for every LocomotionEnv regardless of stand_only, so the true ambient baseline
            # going into tonight's stand-only work was 6.0 all along, not env.py's bare default
            # of 3.0. Caught by actually printing the value before vs. after this assignment
            # instead of trusting the comment above it.
            #
            # Made this a scaffold (HeightRewardScaffoldCallback, train_locomotion.py) instead
            # of a one-off constant edit -- the whole point of tonight's "is it stuck at a low
            # equilibrium" diagnosis was that it only shows up well into training, and finding
            # it required manually stopping, editing a constant, and restarting/warm-starting
            # every time. The scaffold anneals this automatically (6.0 -> HEIGHT_SCALE_TARGET)
            # on every future run without needing another manual round-trip, and (like the
            # other scaffolds) resumes correctly across an --init-from continuation instead of
            # restarting its own ramp from scratch.
        super().__init__(render_mode=render_mode, opponent_policy_path=None, getup_curriculum_prob=0.0)
        self._b_root_x_qpos = self.model.joint("b_root_x").qposadr[0]
        self._b_root_x_dof = self.model.joint("b_root_x").dofadr[0]
        self._a_root_x_dof = self.model.joint("a_root_x").dofadr[0]
        self._zero_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._target_vx = 0.0
        self._knee_r_qpos = self.model.joint("a_knee_r").qposadr[0]
        self._knee_l_qpos = self.model.joint("a_knee_l").qposadr[0]
        self._knee_r_dof = self.model.joint("a_knee_r").dofadr[0]
        self._knee_l_dof = self.model.joint("a_knee_l").dofadr[0]
        self._hip_r_dof = self.model.joint("a_hip_r").dofadr[0]
        self._hip_l_dof = self.model.joint("a_hip_l").dofadr[0]
        # scaffolding (docs 10.37): even a forced-standing, high-target-velocity reset still
        # crouched back to ~-136deg median within 1s (measured on a mid-training v5
        # checkpoint) -- reward-side pressure (10.34/10.35) repeatedly failed to out-argue the
        # policy's own experience that extending is risky. Instead of persuading, remove the
        # escape hatch: during calm episodes only (harsh/collapse episodes still need the full
        # range to recover from genuine collapse), clamp the knee no deeper than this floor.
        # None = no clamp (full range) -- the default until a training run opts in via
        # set_min_knee_deg(), so this class stays a no-op change for anything not using it.
        self._min_knee_deg = None
        self._episode_knee_floor = None  # this episode's floor in radians, or None
        self._is_calm = False
        # velocity assist (docs 10.45): a pure physics check (max root_x thrust, no other
        # joints) showed the actuator has plenty of power -- 1.5 m/s reachable -- but the
        # torso outruns its own base of support and falls by ~6.5s with no leg coordination.
        # Real mean |actual vx| stayed ~0.13-0.17 m/s across v2/v4/v5/v6/v7/v8b regardless of
        # reward changes: likely the same root cause as the knee habit (10.37) -- any
        # committed attempt at speed risks a fall, so the policy converges to barely moving.
        # This is an external qfrc_applied nudge toward target_vx (calm episodes only, same
        # gating as the knee floor) -- like someone lightly pushing you the right way while
        # you find your own coordination, not doing the walking for you. 0.0 = no-op default.
        self._assist_scale = 0.0
        self._stand_streak = 0
        # curriculum stage (2026-10-09): CALM_EPISODE_FRAC was tuned for the combat league
        # (practice recovering mid-fight), never re-examined for stand-only's much narrower
        # goal -- "stand still and hold it" and "get up off the ground" are different skills,
        # and asking for both from the first step may be a harder problem than sequencing them.
        # Instance attribute (not the bare module constant) so set_calm_frac() can anneal it
        # per-run via env_method, same pattern as the knee floor / velocity assist scaffolds.
        self._calm_frac = CALM_EPISODE_FRAC
        self._replay_pool = []
        self._replay_frac = 0.0
        self._replayed = False
        self._start_info = None
        # gear curriculum (strong-body-first): each reset draws a gear multiplier from
        # [1, _gear_max] for the 'a' limb actuators (root_x/root_ry are zeroed in stand_only).
        # _gear_max == 1.0 (default) leaves the model untouched.
        self._gear_max = 1.0
        self._gear_scale = 1.0
        self._gear_idx = [i for i in range(self.model.nu)
                          if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or '').startswith('a_')
                          and 'root' not in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)]
        self.model.actuator_gear[self._gear_idx, 0] *= BODY_GEAR_SCALE   # body spec
        self._gear_base = self.model.actuator_gear[:, 0].copy()          # gear curriculum scales this
        self._a_hinge_limits = []
        for j in range(self.model.njnt):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if (jname and jname.startswith('a_') and 'root' not in jname
                    and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and self.model.jnt_limited[j]):
                lo, hi = self.model.jnt_range[j]
                self._a_hinge_limits.append((int(self.model.jnt_qposadr[j]), float(lo), float(hi)))

        self._a_bodies = [i for i in range(self.model.nbody) if self.model.body(i).name.startswith("a_")]
        self._a_masses = np.array([self.model.body_mass[i] for i in self._a_bodies])
        self._prev_com_x = None
        base_dim = self.observation_space.shape[0]
        # +1 for target_vx, +3 for stand-only balance features (com_rel_x, com_rel_z, com_vel_x --
        # see _obs()). The base observation is just raw joint qpos/qvel, so a stand-only policy
        # has to learn forward kinematics from scratch to even know where its own centre of mass
        # sits relative to its feet -- exactly the thing a hand-coded capture-point controller gets
        # for free by computing it directly (stand-only diagnosis, 2026-10-09: RL plateaued well
        # below even that simple hand controller's ~15-17% survival for 23M+ steps straight,
        # despite reward fixes that should have helped -- handing it the same computed signal
        # the hand controller used is a different kind of lever than reward shaping).
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(base_dim + 4,), dtype=np.float32)

    def set_min_knee_deg(self, deg):
        """deg=None disables the clamp entirely. Called externally (KneeScaffoldCallback)
        to anneal the floor over training -- SB3's env_method() broadcasts this to every
        SubprocVecEnv worker."""
        self._min_knee_deg = None if deg is None else np.deg2rad(deg)

    def set_assist_scale(self, scale):
        """Called externally (VelocityAssistCallback) to anneal the external push-toward-
        target-velocity force. 0.0 disables it entirely."""
        self._assist_scale = scale

    def set_calm_frac(self, frac):
        """Called externally (CalmFracCallback) to anneal the fraction of episodes that start
        calm (near-standing) vs. harsh (random collapse). Same env_method broadcast pattern as
        set_min_knee_deg/set_assist_scale."""
        self._calm_frac = frac

    def set_height_reward_scale(self, scale):
        """Called externally (HeightRewardScaffoldCallback) to anneal env.py's module-global
        head-height reward scale. Mutates the module global directly (there's no per-instance
        copy of it -- env.py's reward calc reads the bare name each step), same as this
        class's __init__ does for BALANCE_KP/STABILITY_REWARD_SCALE, just now adjustable
        mid-run instead of fixed once at construction."""
        env_module.HEAD_HEIGHT_REWARD_SCALE = scale

    def _getup_contacts(self):
        """(foot_down, head_down): any 'a' foot/toe body, resp. the head, touching a non-fighter
        geom (the floor)."""
        m, d = self.model, self.data
        names = getattr(self, '_geom_body_names', None)
        if names is None:
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ''
                     for g in range(m.ngeom)]
            self._geom_body_names = names
        foot = head = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            n1, n2 = names[c.geom1], names[c.geom2]
            for n, o in ((n1, n2), (n2, n1)):
                if n.startswith('a_') and not (o.startswith('a_') or o.startswith('b_')):
                    if n[2:] in _FOOT_BODIES:
                        foot = 1.0
                    elif n[2:] == 'head':
                        head = 1.0
        return foot, head

    def _obs(self):
        com_x = np.sum(self._a_masses * self.data.xipos[self._a_bodies, 0]) / self._a_masses.sum()
        com_z = np.sum(self._a_masses * self.data.xipos[self._a_bodies, 2]) / self._a_masses.sum()
        feet_x = [self.data.site_xpos[s][0] for s in self.a_feet]
        feet_z = [self.data.site_xpos[s][2] for s in self.a_feet]
        com_rel_x = com_x - float(np.mean(feet_x))   # horizontal COM offset from foot centre
        com_rel_z = com_z - float(min(feet_z))        # COM height above the lower foot
        # finite-difference COM velocity (matches the hand capture-point controller's own
        # approach) -- 0.0 on the very first observation of an episode, same as that controller.
        com_vel_x = 0.0 if self._prev_com_x is None else (com_x - self._prev_com_x) / 0.02
        self._prev_com_x = com_x
        extra = np.array([com_rel_x, com_rel_z, com_vel_x], dtype=np.float32)
        return np.concatenate([super()._obs(), [self._target_vx], extra]).astype(np.float32)

    def _resample_target_vx(self):
        self._target_vx = float(self.np_random.uniform(-TARGET_VX_RANGE, TARGET_VX_RANGE))

    def _randomize_a_state(self, severity_lo, severity_hi):
        """Broader than Fighter2DEnv._collapse_pose: interpolates between the current
        (near-standing) pose and a widely-sampled target by a random severity in
        [0, severity_hi], covering everything from a light perturbation to a full collapse
        in one curriculum instead of a binary standing/collapsed split. Includes arms and
        torso pitch too, not just legs/waist -- a real stagger disturbs the whole body, not
        just the base."""
        severity = float(self.np_random.uniform(severity_lo, severity_hi))
        legs_waist = (("hip_r", 0.6), ("hip_l", 0.6), ("waist", 0.6))
        for j, frac in legs_waist:
            qpos = self.model.joint(f"a_{j}").qposadr[0]
            lo, hi = self.model.joint(f"a_{j}").range
            target = self.np_random.uniform(lo * frac, hi * frac)
            self.data.qpos[qpos] = severity * target + (1.0 - severity) * self.data.qpos[qpos]
        for j in ("knee_r", "knee_l"):
            qpos = self.model.joint(f"a_{j}").qposadr[0]
            lo, _hi = self.model.joint(f"a_{j}").range
            target = self.np_random.uniform(lo * 0.7, 0.0)
            self.data.qpos[qpos] = severity * target + (1.0 - severity) * self.data.qpos[qpos]
        for j in ("ankle_r", "ankle_l", "shoulder_r", "shoulder_l", "elbow_r", "elbow_l"):
            qpos = self.model.joint(f"a_{j}").qposadr[0]
            lo, hi = self.model.joint(f"a_{j}").range
            target = self.np_random.uniform(lo, hi)
            self.data.qpos[qpos] = severity * target + (1.0 - severity) * self.data.qpos[qpos]
        self.data.qpos[self.a_ry_qpos] += severity * self.np_random.uniform(-0.6, 0.6)
        return severity

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._gear_scale = 1.0 if self._gear_max <= 1.0 else float(self.np_random.uniform(1.0, self._gear_max))
        self.model.actuator_gear[self._gear_idx, 0] = self._gear_base[self._gear_idx] * self._gear_scale

        self.data.qpos[self._b_root_x_qpos] = B_PARK_X
        # Two-tier curriculum (docs 10.36): measured that falls happen mostly during
        # knee-extended moments, not crouched ones (knee ~-52deg median 1s before a fall vs
        # ~-138deg overall) -- the policy has correctly learned, from its OWN experience,
        # that standing/extending is genuinely riskier under a curriculum where every single
        # episode starts with some random collapse severity and a 40% chance of a shove.
        # Reward-scale bumps (10.34/10.35) couldn't out-argue that real experience. Instead,
        # give it a lot more practice at "nothing threatening is happening, standing is
        # fine" by making most episodes calm (low severity, no push) and only a minority
        # keep the full hard curriculum -- closer to how a real fighter is only crouched
        # defensively some of the time, not permanently braced.
        is_calm = self.np_random.random() < self._calm_frac
        severity = self._randomize_a_state(0.0 if is_calm else HARSH_SEVERITY_LO,
                                            CALM_SEVERITY_HI if is_calm else 1.0)
        # only calm episodes get the knee floor -- harsh/collapse episodes need the full
        # range to actually recover from a genuine collapse (docs 10.37)
        self._episode_knee_floor = self._min_knee_deg if is_calm else None
        self._is_calm = is_calm
        self._stand_streak = 0
        self._prev_com_x = None
        mujoco.mj_forward(self.model, self.data)

        saved_ctrl = self.data.ctrl.copy()
        self.data.ctrl[:] = 0.0
        for _ in range(int(GETUP_SETTLE_STEPS * severity)):
            mujoco.mj_step(self.model, self.data)
        self.data.ctrl[:] = saved_ctrl

        # push probability follows the same calm/harsh split -- a calm episode should mean
        # genuinely no threat, not just a smaller starting bend.
        # docs 10.54: tried raising this to 0.7 (v13d) and 0.45 (v13e), both warm-started from
        # v13c -- both improved calm and the push sub-case itself, but both regressed the
        # plain-collapse (no-push) case back to ~75% (from v13c's 47.5%), regardless of how
        # much the probability moved. Root cause: push/no-push split the SAME harsh-episode
        # budget, so raising push_prob necessarily steals training volume from no-push -- a
        # zero-sum tradeoff within the harsh branch, not a volume-of-exposure problem solvable
        # by tuning this constant. Reverted to the original 0.4 (the v13c value, still the best
        # weighted result found: calm 22.5%, push 67.5%, no-push 47.5%). The push sub-case
        # itself remains unsolved -- would need a different lever (e.g. its own scaffold)
        # rather than this split ratio.
        push_prob = 0.0 if is_calm else 0.4
        if self.np_random.random() < push_prob:
            self.data.qvel[self._a_root_x_dof] += self.np_random.uniform(-2.5, 2.5)

        # failed-start replay (opt-in via set_replay_pool; empty pool = no behaviour change)
        self._replayed = False
        if self._replay_pool and self.np_random.random() < self._replay_frac:
            self._load_replay_state(self._replay_pool[int(self.np_random.integers(len(self._replay_pool)))])
        self._record_start_state()
        self._prev_phi = min(self.data.xpos[self.a_head_id][2], HEAD_PROGRESS_CAP) / HEAD_PROGRESS_CAP
        self._archive_cool = 0
        self._head_best = min(self.data.xpos[self.a_head_id][2], HEAD_RECORD_CAP) / HEAD_RECORD_CAP
        self._start_head = float(self.data.xpos[self.a_head_id][2])

        self._resample_target_vx()
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), info

    def set_replay_pool(self, pool, frac):
        """pool: list of (qpos, qvel, is_calm) start states from episodes the policy failed, kept
        only for clusters of intermediate difficulty (ReplayCurriculumCallback). Each reset then
        loads one of them with probability `frac`. Same env_method broadcast as set_calm_frac."""
        self._replay_pool = list(pool)
        self._replay_frac = float(frac)

    def set_archive_collect(self, flag):
        """Report high-head states reached from floor starts in info['archive_state'] (see ARCHIVE_*)."""
        self._collect_archive = bool(flag)

    def set_gear_scale_max(self, scale_max):
        """Upper end of the per-episode gear multiplier range (1.0 = nominal body only)."""
        self._gear_max = float(scale_max)

    def _load_replay_state(self, entry):
        qpos, qvel, calm = entry
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        # small joint jitter so the policy can't just memorise a handful of exact states
        for adr, lo, hi in self._a_hinge_limits:
            self.data.qpos[adr] = np.clip(self.data.qpos[adr] + self.np_random.normal(0.0, REPLAY_JITTER_RAD), lo, hi)
        self._episode_knee_floor = self._min_knee_deg if calm else None
        self._is_calm = calm
        self._stand_streak = 0
        self._replayed = True
        mujoco.mj_forward(self.model, self.data)

    def _record_start_state(self):
        """Snapshot of the state this episode starts from; step() attaches it to the terminal info."""
        self._start_info = dict(
            qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(), calm=bool(self._is_calm),
            knee=float(np.degrees(min(self.data.qpos[self._knee_r_qpos], self.data.qpos[self._knee_l_qpos]))),
            z=float(self.data.xpos[self.a_torso_id][2]), replayed=self._replayed, gear=self._gear_scale)

    def step(self, action):
        if self._stand_only:
            # the horizontal thrust actuator (root_x) isn't a leg or arm -- a real body can't push
            # itself sideways from nothing, and the trained policy saturated it on ~half of steps
            action = np.array(action, dtype=np.float32, copy=True)
            action[0] = 0.0
        if self._is_calm and self._assist_scale and not self._stand_only:
            self.data.qfrc_applied[self._a_root_x_dof] = self._assist_scale * self._target_vx
        else:
            self.data.qfrc_applied[self._a_root_x_dof] = 0.0

        obs, reward, terminated, truncated, info = super().step(action, b_full_action=self._zero_action)
        if self._knee_gate:
            head_z = self.data.xpos[self.a_head_id][2]
            kg = min(1.0, max(0.0, (head_z - KNEE_GATE_HEAD_LO) / (KNEE_GATE_HEAD_HI - KNEE_GATE_HEAD_LO)))
            bd = info["reward_breakdown"]
            reward -= bd["knee_avoid"] * (1.0 - kg)   # reward is the plain sum of these terms (env.py)
            bd["knee_avoid"] *= kg
        if self._head_progress > 0.0:
            phi = min(self.data.xpos[self.a_head_id][2], HEAD_PROGRESS_CAP) / HEAD_PROGRESS_CAP
            bonus = self._head_progress * (HEAD_PROGRESS_GAMMA * phi - self._prev_phi)
            self._prev_phi = phi
            reward += bonus
            info["reward_breakdown"]["head_progress"] = bonus
        if self._head_record > 0.0:
            phi_r = min(self.data.xpos[self.a_head_id][2], HEAD_RECORD_CAP) / HEAD_RECORD_CAP
            if phi_r > self._head_best:
                foot_down, _ = self._getup_contacts()
                if foot_down:
                    bonus_r = self._head_record * (phi_r - self._head_best)
                    self._head_best = phi_r
                    reward += bonus_r
                    info["reward_breakdown"]["head_record"] = bonus_r
        if self._collect_archive:
            self._archive_cool -= 1
            hz_a = self.data.xpos[self.a_head_id][2]
            if (self._archive_cool <= 0 and self._start_info is not None
                    and (self._start_info["z"] < 0.45 or self._replayed)
                    and ARCHIVE_HEAD_LO <= hz_a < ARCHIVE_HEAD_HI
                    and hz_a >= self._start_head + ARCHIVE_GAIN):
                foot_a, _ = self._getup_contacts()
                pitch_a = (self.data.qpos[self.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi
                if foot_a and abs(pitch_a) < np.deg2rad(70.0):
                    info["archive_state"] = (self.data.qpos.copy(), self.data.qvel.copy(), float(hz_a))
                    self._archive_cool = ARCHIVE_COOLDOWN
        # 'b' is scripted to be irrelevant, but down_steps['b'] can still creep up over time
        # (no actuation holding it up) and trip a_out's `a_out or b_out` termination even
        # though 'a' itself is fine -- only 'a' failing to recover should end an episode here.
        if info.get("b_out") and not info.get("a_out"):
            terminated = False
        self.data.qpos[self._b_root_x_qpos] = B_PARK_X
        self.data.qvel[self._b_root_x_dof] = 0.0

        if self._stand_only:
            # "stance"/"contact_stance" are inherited unchanged from the combat reward --
            # they directly pay for a WIDER front-back foot gap (up to STANCE_GAP_REF=0.4m),
            # meant to keep a fighter's footwork from narrowing into an easily-swept stance.
            # Never zeroed for stand_only like velocity/gait were, so a stand-only policy was
            # quietly being paid to splay its legs apart regardless of whether that helped
            # balance at all (flagged from watching v6: legs flung into a wide front-back
            # split well past what COM/brace alone would ask for). Zero both out here --
            # com_reward is the one that should decide how the feet are placed now.
            stance_total = (info["reward_breakdown"].get("stance", 0.0)
                             + info["reward_breakdown"].get("contact_stance", 0.0))
            reward -= stance_total
            info["reward_breakdown"]["stance"] = 0.0
            info["reward_breakdown"]["contact_stance"] = 0.0

            # same bug, different term (2026-10-08): "recovery" rewards upward torso velocity
            # gated by max(0, 1 - head_z/STANDING_HEAD_HEIGHT) -- STANDING_HEAD_HEIGHT is a
            # fully-upright reference (1.29), so that gate stays substantially open (measured
            # ~0.25 of its max) even in a relaxed stance that comfortably clears FALL_HEIGHT.
            # Meant to pay for genuinely getting up off the ground, it was quietly also paying
            # for any small upward bob while already standing -- standing still earns it
            # nothing, bobbing earns it repeatedly, which likely explains the "tap-dancing"
            # balance behavior watched on v12. Zero it out once already above FALL_HEIGHT, so
            # it only ever fires while genuinely down and makes the per-step reward while
            # standing come entirely from the "stay up" terms (height/stability/com/stand_streak).
            if self.data.xpos[self.a_torso_id][2] >= FALL_HEIGHT:
                reward -= info["reward_breakdown"].get("recovery", 0.0)
                info["reward_breakdown"]["recovery"] = 0.0

        if self._episode_knee_floor is not None:
            clamped = False
            for qadr, dadr in ((self._knee_r_qpos, self._knee_r_dof), (self._knee_l_qpos, self._knee_l_dof)):
                if self.data.qpos[qadr] < self._episode_knee_floor:
                    self.data.qpos[qadr] = self._episode_knee_floor
                    self.data.qvel[dadr] = max(0.0, self.data.qvel[dadr])  # keep any extending motion
                    clamped = True
            if clamped:
                mujoco.mj_forward(self.model, self.data)

        actual_vx = self.data.qvel[self._a_root_x_dof]
        velocity_credit = np.exp(-((actual_vx - self._target_vx) ** 2) / (2.0 * VELOCITY_SIGMA ** 2))
        velocity_reward = 0.0 if self._stand_only else VELOCITY_REWARD_SCALE * velocity_credit
        hip_r_vel = self.data.qvel[self._hip_r_dof]
        hip_l_vel = self.data.qvel[self._hip_l_dof]
        # docs 10.47: v10 measured r_gait climbing hard while mean|actual vx| and its target
        # correlation both got WORSE -- the anti-phase condition alone was satisfiable by
        # marching/scissoring in place, unrelated to whether the torso was actually moving.
        # Gating by velocity_credit (the same match-quality term velocity_reward already
        # uses) closes that loophole directly: this can only pay out while ALSO tracking
        # target_vx reasonably well, so marching against a real (nonzero) target now earns
        # both terms' worth of nothing instead of gait_reward alone for free.
        gait_raw = max(0.0, -(hip_r_vel * hip_l_vel))
        gait_reward = 0.0 if self._stand_only else GAIT_REWARD_SCALE * (1.0 - np.exp(-gait_raw / GAIT_REF)) * velocity_credit
        vz_down = -self.data.qvel[self.a_z_dof]
        fall_gate = min(1.0, max(0.0, vz_down / BRACE_VZ_REF))
        knee_mean = 0.5 * (self.data.qpos[self._knee_r_qpos] + self.data.qpos[self._knee_l_qpos])
        knee_straight = min(1.0, max(0.0, (knee_mean + np.deg2rad(140.0)) / np.deg2rad(140.0)))
        brace_reward = BRACE_REWARD_SCALE * fall_gate * knee_straight
        # knee_straight above uses the MEAN of both knees -- fine for brace (a same-instant
        # reflex where partial credit for a half-hearted brace is reasonable), but measured on
        # a real checkpoint that the ungated version below was being satisfied by straightening
        # ONE knee to ~0deg while leaving the other pinned at -140 (mean looks like "both knees
        # half-bent", same score, torso stays just as low). knee_avoid right above already hit
        # this exact masking problem once and fixed it by using MIN instead of the average --
        # same fix here, since the whole point is both legs need to extend to actually raise
        # the torso.
        # recovery_gate (2026-10-10): knee_straight/torso_upright were tuned and validated
        # entirely during calm_frac=1.0 training (near-standing starts only), where "straighten
        # the knees, stay upright" never conflicts with anything. Once harsh/collapse starts got
        # mixed in (CurriculumGateCallback phase 2), the harsh-only gate measured at 8/100 vs.
        # ~30% calm -- a real human getting up off the ground bends the knees to tuck the feet
        # and pitches the torso forward/down to push up through hands/knees; both rewards were
        # penalizing exactly that necessary intermediate posture on every single step, fighting
        # the only known way to actually stand up from flat. Gated the same way recovery_reward
        # (env.py) already was -- ramps 0 at the ground up to full strength at FALL_HEIGHT, so
        # calm/already-standing behavior (the thing they were tuned against) is unaffected.
        #
        # First version used a pure linear ratio (torso_z / FALL_HEIGHT), which never actually
        # reaches 0 -- a genuinely lying-flat torso still sits at ~0.21m (body thickness), giving
        # ratio~0.30. Measured directly (same lying pose, knees swept bent->straight): straight
        # still won on total reward (2.188 vs 1.785 bent) because knee_straight's residual
        # (KNEE_STRAIGHT_SCALE * 0.30 ~= 0.9) outweighed RECOVERY_REWARD_SCALE's realized value
        # (0.441) -- the softer gate wasn't soft enough to stop favoring "lie flat and extend"
        # over "tuck and push". Hard-floored instead: zero below half of FALL_HEIGHT, linear
        # ramp to full strength only in the top half -- a truly flat posture now gets exactly 0
        # from these two terms, leaving recovery/height as the only signal while on the ground.
        recovery_gate = min(1.0, max(0.0, (self.data.xpos[self.a_torso_id][2] / FALL_HEIGHT - 0.5) / 0.5))
        knee_min = min(self.data.qpos[self._knee_r_qpos], self.data.qpos[self._knee_l_qpos])
        knee_straight_min = min(1.0, max(0.0, (knee_min + np.deg2rad(140.0)) / np.deg2rad(140.0)))
        knee_straight_reward = (KNEE_STRAIGHT_SCALE * knee_straight_min * recovery_gate
                                 if self._stand_only else 0.0)
        torso_upright_reward = 0.0
        if self._stand_only:
            a_ry = self.data.qpos[self.a_ry_qpos]
            upright_ratio = min(1.0, max(0.0, 1.0 - abs(a_ry) / TORSO_ANGLE_REF))
            torso_upright_reward = TORSO_UPRIGHT_SCALE * upright_ratio ** 2 * recovery_gate
        com_reward = 0.0
        if self._stand_only:
            com_x = np.sum(self._a_masses * self.data.xipos[self._a_bodies, 0]) / self._a_masses.sum()
            feet_x = [self.data.site_xpos[s][0] for s in self.a_feet]
            lo, hi = min(feet_x) - 0.12, max(feet_x) + 0.12
            margin = min(com_x - lo, hi - com_x)
            com_reward = COM_REWARD_SCALE * min(1.0, max(0.0, margin / COM_MARGIN_REF))
            if self._ground_shaping:
                com_reward *= recovery_gate
        ground_height_reward = 0.0
        getup_shift_reward = 0.0
        getup_penalty = 0.0
        upright_ok = True
        if self._ground_shaping:
            head_z = self.data.xpos[self.a_head_id][2]
            ground_height_reward = GROUND_HEIGHT_SCALE * min(1.0, max(0.0, head_z / GROUND_HEIGHT_REF))
        if self._getup:
            foot_down, head_down = self._getup_contacts()
            ground_height_reward *= foot_down
            torso_z = self.data.xpos[self.a_torso_id][2]
            if foot_down and torso_z < GETUP_SHIFT_MAX_Z:
                dx = abs(com_x - np.mean(feet_x))
                getup_shift_reward = GETUP_SHIFT_SCALE * min(1.0, max(0.0, 1.0 - dx / GETUP_SHIFT_REF))
            pitch = self.data.qpos[self.a_ry_qpos]
            wrapped = (pitch + np.pi) % (2 * np.pi) - np.pi
            getup_penalty = (GETUP_HEAD_CONTACT_PEN * head_down
                             + GETUP_INVERT_PEN * float(abs(wrapped) > np.deg2rad(GETUP_INVERT_DEG)))
            upright_ok = bool(abs(wrapped) < np.deg2rad(GETUP_UPRIGHT_DEG) and head_z >= GETUP_UPRIGHT_HEAD_Z)
        stand_streak_reward = 0.0
        if self._stand_only:
            if self.data.xpos[self.a_torso_id][2] >= FALL_HEIGHT and upright_ok:
                self._stand_streak += 1
            else:
                self._stand_streak = 0
            stand_streak_reward = STAND_STREAK_SCALE * min(1.0, self._stand_streak / STAND_STREAK_REF)
            info["stand_streak_steps"] = self._stand_streak  # raw step count, uncapped -- for
                                                              # curriculum gating (CurriculumGateCallback)
        reward = (reward + velocity_reward + gait_reward + brace_reward + com_reward
                  + stand_streak_reward + knee_straight_reward + torso_upright_reward
                  + ground_height_reward + getup_shift_reward - getup_penalty)
        info["reward_breakdown"]["getup_shift"] = getup_shift_reward
        info["reward_breakdown"]["getup_pen"] = -getup_penalty
        info["reward_breakdown"]["velocity"] = velocity_reward
        info["reward_breakdown"]["gait"] = gait_reward
        info["reward_breakdown"]["brace"] = brace_reward
        info["reward_breakdown"]["com"] = com_reward
        info["reward_breakdown"]["stand_streak"] = stand_streak_reward
        info["reward_breakdown"]["knee_straight"] = knee_straight_reward
        info["reward_breakdown"]["torso_upright"] = torso_upright_reward
        info["reward_breakdown"]["ground_height"] = ground_height_reward
        info["target_vx"] = self._target_vx
        info["torso_z"] = float(self.data.xpos[self.a_torso_id][2])
        if (terminated or truncated) and self._start_info is not None:
            info["episode_start"] = self._start_info

        if RESAMPLE_EVERY_STEPS is not None and self.step_count % RESAMPLE_EVERY_STEPS == 0:
            self._resample_target_vx()

        return self._obs(), reward, terminated, truncated, info
