"""Trains the solo mobility/recovery policy (locomotion_env.LocomotionEnv) -- no opponent,
no --side, no --opponent-from. See locomotion_env.py for what this is testing.

Usage: train_locomotion.py [--timesteps N] [--n-envs N] [--init-from CKPT] [--out NAME]
"""
import argparse
import pathlib
import time
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from locomotion_env import CALM_EPISODE_FRAC, LocomotionEnv, STAND_STREAK_REF

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"

# same clamp as train_league.py (docs 10.20) -- keeps exploration alive without needing an
# entropy bonus (which overshot into runaway growth the one time it was tried).
LOG_STD_MIN = np.log(0.05)
LOG_STD_MAX = np.log(0.6)


class BreakdownCallback(BaseCallback):
    KEYS = ["height", "stability", "stance", "knee_avoid", "recovery", "jerk", "velocity", "gait", "brace", "com",
            "stand_streak", "knee_straight", "torso_upright", "ground_height", "getup_shift", "getup_pen", "head_progress", "head_record", "speed_pen", "air_pen", "standready_dense", "settle"]

    def __init__(self, window=100):
        super().__init__()
        self._running = None
        self._recent = {k: deque(maxlen=window) for k in self.KEYS}

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._running = {k: np.zeros(n) for k in self.KEYS}

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            bd = info.get("reward_breakdown")
            if not bd:
                continue
            for k in self.KEYS:
                self._running[k][i] += bd.get(k, 0.0)
            if self.locals["dones"][i]:
                for k in self.KEYS:
                    self._recent[k].append(self._running[k][i])
                    self._running[k][i] = 0.0
        for k in self.KEYS:
            if self._recent[k]:
                self.logger.record(f"rollout/r_{k}", float(np.mean(self._recent[k])))
        return True


class LogStdClampCallback(BaseCallback):
    """Periodic version of the once-at-the-end clamp below -- a v3fresh run (docs 10.34)
    regressed hard: fair-condition fall rate 0%->12.5% and worst-case 35%->55.3% over its
    last ~9M steps, with std quietly narrowing the whole time (0.98->0.57) and no mid-run
    floor/ceiling to stop it, unlike train_league.py's per-round clamp. Clamping every
    `freq` steps instead of only in the `finally` block keeps a minimum exploration floor
    alive throughout training, not just in the final saved snapshot."""

    def __init__(self, freq=50_000):
        super().__init__()
        self.freq = freq
        self._last_clamp_step = 0

    def _on_step(self):
        if self.num_timesteps - self._last_clamp_step >= self.freq:
            with torch.no_grad():
                self.model.policy.log_std.data.clamp_(min=LOG_STD_MIN, max=LOG_STD_MAX)
            self._last_clamp_step = self.num_timesteps
        return True


class LinearScaffoldCallback(BaseCallback):
    """Shared machinery for annealing a LocomotionEnv scaffold parameter from `start_val` to
    `end_val` over `relax_frac` of THIS CALL's own timestep budget, then holding at `end_val`
    for good -- used by both KneeScaffoldCallback (docs 10.37/10.38) and
    VelocityAssistCallback (docs 10.45). Subclass and implement `apply(value)` to broadcast
    the annealed value (via env_method) and pick a `log_key` for the logger.

    docs 10.44/10.45: originally computed progress as num_timesteps / (relax_frac *
    total_timesteps) -- num_timesteps is the model's CUMULATIVE step count (carried over by a
    warm start), so on any --init-from continuation this either (a) massively overshot the
    denominator for an already-relaxed parameter being continued, re-loosening a scaffold that
    had already finished relaxing (the exact v8b->v9 knee bug), or (b) for a BRAND NEW
    parameter added mid-project (velocity assist), the huge inherited num_timesteps instantly
    saturated frac to 1.0, skipping the ramp before a single step of this run had happened.
    Both were the same root mistake: measuring progress against absolute step count instead of
    this call's own elapsed steps. Fixed by tracking elapsed = num_timesteps - (num_timesteps
    at the start of THIS training_start), i.e. always 0 at the beginning of any given call
    regardless of warm-start history. already_relaxed=True skips straight to end_val from the
    first step -- pass it explicitly on a continuation of a schedule that already finished
    ramping in a prior run, instead of trying to re-derive an absolute step count."""

    log_key = "rollout/scaffold_value"

    def __init__(self, total_timesteps, start_val, end_val, relax_frac=0.6, freq=50_000,
                 already_relaxed=False):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.start_val = start_val
        self.end_val = end_val
        self.relax_frac = relax_frac
        self.freq = freq
        self.already_relaxed = already_relaxed
        self._start_step = None
        self._last_set_step = -freq  # force a set on the very first _on_step

    def apply(self, value):
        raise NotImplementedError

    def _on_training_start(self):
        self._start_step = self.num_timesteps

    def _on_step(self):
        if self.num_timesteps - self._last_set_step >= self.freq:
            if self.already_relaxed:
                frac = 1.0
            else:
                elapsed = self.num_timesteps - self._start_step
                frac = min(1.0, elapsed / (self.relax_frac * self.total_timesteps))
            value = self.start_val + (self.end_val - self.start_val) * frac
            self.apply(value)
            self.logger.record(self.log_key, value)
            self._last_set_step = self.num_timesteps
        return True


class KneeScaffoldCallback(LinearScaffoldCallback):
    """Anneals LocomotionEnv's per-calm-episode knee floor (docs 10.37) from `start_deg`
    (restrictive -- physically can't retreat into a deep crouch) to `end_deg`, then holds.

    v6 (docs 10.38) relaxed all the way to the joint's real limit (-140, fully unconstrained)
    and regressed hard as it did -- knee-straighter-than-90deg dropped 27%->12.5% and mean
    |vx| dropped 0.174->0.131 m/s between the 3.75M and 9.25M checkpoints, tracking the floor
    loosening from -78 to -119. Calm episodes are explicitly the "nothing threatening is
    happening" case -- there's no real reason they'd ever need a full-depth crouch (that's
    what harsh/collapse episodes are for, and they keep their own unconstrained range
    regardless of this schedule). So end_deg now stops at a permanent partial floor instead
    of releasing back to the full range -- matches the standard curriculum-RL mitigation for
    this exact failure mode (mix difficulty levels / cap the relaxation rather than fully
    removing scaffolding, since fully removing it invites reverting to the pre-scaffold
    policy)."""

    log_key = "rollout/knee_floor_deg"

    def __init__(self, total_timesteps, start_deg=-50.0, end_deg=-80.0, **kwargs):
        super().__init__(total_timesteps, start_deg, end_deg, **kwargs)

    def apply(self, value):
        self.training_env.env_method("set_min_knee_deg", value)


class VelocityAssistCallback(LinearScaffoldCallback):
    """Anneals LocomotionEnv's external push-toward-target-velocity force (docs 10.45) from
    `start_scale` down to a small permanent `end_scale` (not 0 -- the knee scaffold's lesson
    from 10.38/10.40 was that fully removing a scaffold invites reverting).

    Measured mean |actual vx| stuck at ~0.13-0.17 m/s across five straight reward/curriculum
    changes (v2 through v8b), while a pure-physics check (max root_x thrust, no other joints)
    showed the actuator can reach 1.5 m/s alone -- it's not an actuator power problem, it's
    that any real attempt at speed risks falling with no leg coordination yet, so the policy
    converges to barely moving (same shape of problem as the knee habit). The assist is an
    external nudge in qfrc_applied proportional to target_vx, calm episodes only -- gives the
    policy sustained experience of being in motion (and what balance recovery looks like at
    speed) without asking it to generate 100% of the propulsion from scratch."""

    log_key = "rollout/assist_scale"

    def __init__(self, total_timesteps, start_scale=30.0, end_scale=8.0, **kwargs):
        super().__init__(total_timesteps, start_scale, end_scale, **kwargs)

    def apply(self, value):
        self.training_env.env_method("set_assist_scale", value)


class HeightRewardScaffoldCallback(LinearScaffoldCallback):
    """Anneals the stand-only head-height reward scale from `start_scale` (6.0 -- the ambient
    baseline every LocomotionEnv gets from a module-level override left by an earlier
    locomotion-mode experiment, docs 10.33-10.35, NOT env.py's bare 3.0 default) up to
    `end_scale`, instead of a one-off manual constant edit.

    Measured directly (2026-10-09, qfrc_bias at a hand-set upright pose) that holding full leg
    extension against gravity costs only ~1-5% of each joint's max torque -- not a capability
    problem. Yet checkpoints kept settling into a stable but LOW equilibrium (torso_z
    0.10-0.63, std~0 -- genuinely stable, not falling) even with a dedicated knee-straightness
    reward added. The fix (raising this scale so "stand tall" clearly beats "sit low and
    stable" in the policy's own value estimate) only showed its need well into a training run,
    and required manually stopping, editing a constant, and warm-starting every time it came
    up. Baked in as a scaffold instead so every future run gets it automatically and (like the
    other scaffolds) resumes its own ramp correctly across an --init-from continuation."""

    log_key = "rollout/height_reward_scale"

    def __init__(self, total_timesteps, start_scale=6.0, end_scale=10.0, **kwargs):
        super().__init__(total_timesteps, start_scale, end_scale, **kwargs)

    def apply(self, value):
        self.training_env.env_method("set_height_reward_scale", value)


class CurriculumGateCallback(LinearScaffoldCallback):
    """Advances LocomotionEnv's calm/harsh episode-start curriculum (docs 2026-10-09) from
    phase 1 (100% calm -- just learn to hold a stand) to phase 2 (the original combat-tuned
    CALM_EPISODE_FRAC mix, harsh/collapse starts included) only once phase 1 actually looks
    solved, instead of on a fixed timer. A fixed timer can't tell whether phase 1 finished
    early (wasting steps waiting) or is still unsolved when the deadline hits (forcing harsh
    starts onto a policy that hasn't nailed the easy case yet) -- gating on real performance
    avoids both. Trigger: the training-log fall_rate drops to <= FALL_RATE_GATE AND the mean
    longest continuous stand per episode reaches >= STREAK_GATE_STEPS (10s), both sustained
    over `window` consecutive episodes. Once triggered, phase 2 is still phased in gradually
    (the usual LinearScaffoldCallback anneal, starting from the trigger point) rather than
    switched on instantly, to soften the transition.

    "Stand still and hold it" and "get up off the ground" are different skills (the latter
    still has its own dedicated `recovery` reward term) -- this doesn't claim the second is
    free once the first works, just that learning it staged may be easier than both at once."""

    log_key = "rollout/calm_frac"
    FALL_RATE_GATE = 0.1
    STREAK_GATE_STEPS = 500  # 10s at 0.02s/step

    def __init__(self, total_timesteps, end_frac=CALM_EPISODE_FRAC, window=200,
                 post_trigger_relax_frac=0.15, fall_rate_gate=None, streak_gate_steps=None,
                 **kwargs):
        super().__init__(total_timesteps, 1.0, end_frac, relax_frac=post_trigger_relax_frac, **kwargs)
        if fall_rate_gate is not None:
            self.FALL_RATE_GATE = fall_rate_gate
        if streak_gate_steps is not None:
            self.STREAK_GATE_STEPS = streak_gate_steps
        self._fall_recent = deque(maxlen=window)
        self._streak_recent = deque(maxlen=window)
        self._streak_running = None
        self._triggered = False

    def apply(self, value):
        self.training_env.env_method("set_calm_frac", value)

    def _on_training_start(self):
        super()._on_training_start()
        self._streak_running = np.zeros(self.training_env.num_envs)
        if self.already_relaxed:
            self._triggered = True  # continuation of a run whose phase-1 gate already passed

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            self._streak_running[i] = max(self._streak_running[i], info.get("stand_streak_steps", 0))
            if self.locals["dones"][i]:
                self._fall_recent.append(1.0 if info.get("a_out") else 0.0)
                self._streak_recent.append(self._streak_running[i])
                self._streak_running[i] = 0.0
        if self._streak_recent:
            self.logger.record("rollout/max_stand_streak_steps", float(np.mean(self._streak_recent)))

        if not self._triggered and len(self._fall_recent) >= self._fall_recent.maxlen:
            if (np.mean(self._fall_recent) <= self.FALL_RATE_GATE
                    and np.mean(self._streak_recent) >= self.STREAK_GATE_STEPS):
                self._triggered = True
                self._start_step = self.num_timesteps  # the anneal's own clock starts now
                print(f"[curriculum] phase-1 gate passed at step {self.num_timesteps} "
                      f"(fall_rate={np.mean(self._fall_recent):.3f}, "
                      f"streak={np.mean(self._streak_recent):.0f}) -- starting phase-2 anneal",
                      flush=True)

        if self._triggered:
            return super()._on_step()
        if self.num_timesteps - self._last_set_step >= self.freq:
            self.apply(self.start_val)
            self.logger.record(self.log_key, self.start_val)
            self._last_set_step = self.num_timesteps
        return True


class ReplayCurriculumCallback(BaseCallback):
    """Failed-start replay with an intermediate-difficulty filter (a cheap reverse curriculum).

    Every finished episode reports the state it started from (LocomotionEnv puts it in
    info["episode_start"]) and whether it held a 3s continuous stand. Starts are bucketed by
    (initial knee bend, initial torso height) -- the two things that separated winners from
    losers in the calm measurements. Per bucket we keep a rolling success rate and the start
    states of the episodes that FAILED. Every `sync_every` steps the pool handed to the envs is
    the failed starts from buckets whose success rate is in [lo, hi]: buckets at ~0% are
    skipped (nothing learnable there yet -- that's the harsh problem again) and so are buckets
    that are already solved. As a bucket improves past `hi` it graduates out and the pool
    drifts toward harder starts on its own. A fraction `frac` of resets load from the pool; the
    rest keep the normal distribution so easy/upright starts aren't forgotten."""

    KNEE_EDGES = [-90.0, -60.0, -40.0, -25.0, -10.0]       # deg, min knee at episode start
    Z_EDGES = [0.9, 1.0, 1.08, 1.12, 1.16]                  # m, torso z at episode start

    def __init__(self, frac=0.3, sync_every=50_000, lo=0.10, hi=0.70, min_n=20, window=40,
                 per_bucket_cap=200, max_pool=3000):
        super().__init__()
        self.frac, self.sync_every, self.lo, self.hi = frac, sync_every, lo, hi
        self.min_n, self.window, self.cap, self.max_pool = min_n, window, per_bucket_cap, max_pool
        self._outcomes = {}   # bucket -> deque of 0/1 success
        self._fails = {}      # bucket -> deque of (qpos, qvel, calm) of failed, non-replayed starts
        self._recent_replayed = deque(maxlen=200)
        self._recent_fresh = deque(maxlen=200)
        self._streak = None
        self._last_sync = 0

    def _on_training_start(self):
        self._streak = np.zeros(self.training_env.num_envs)
        self._last_sync = self.num_timesteps
        self.training_env.env_method("set_replay_pool", [], self.frac)

    def _bucket(self, start):
        return (int(np.digitize(start["knee"], self.KNEE_EDGES)), int(np.digitize(start["z"], self.Z_EDGES)))

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            self._streak[i] = max(self._streak[i], info.get("stand_streak_steps", 0))
            if not self.locals["dones"][i]:
                continue
            start = info.get("episode_start")
            ok = self._streak[i] >= STAND_STREAK_REF
            self._streak[i] = 0.0
            if start is None:
                continue
            b = self._bucket(start)
            self._outcomes.setdefault(b, deque(maxlen=self.window)).append(1.0 if ok else 0.0)
            (self._recent_replayed if start["replayed"] else self._recent_fresh).append(1.0 if ok else 0.0)
            if not ok and not start["replayed"]:
                self._fails.setdefault(b, deque(maxlen=self.cap)).append((start["qpos"], start["qvel"], start["calm"]))
        if self.num_timesteps - self._last_sync >= self.sync_every:
            self._sync()
        return True

    def _sync(self):
        pool, used = [], 0
        for b, out in self._outcomes.items():
            if len(out) >= self.min_n and self.lo <= np.mean(out) <= self.hi and self._fails.get(b):
                pool.extend(self._fails[b]); used += 1
        if len(pool) > self.max_pool:
            idx = np.random.default_rng(self.num_timesteps).choice(len(pool), self.max_pool, replace=False)
            pool = [pool[j] for j in idx]
        self.training_env.env_method("set_replay_pool", pool, self.frac)
        self.logger.record("replay/pool_size", len(pool))
        self.logger.record("replay/buckets_in_band", used)
        self.logger.record("replay/buckets_seen", len(self._outcomes))
        if self._recent_replayed:
            self.logger.record("replay/success_replayed", float(np.mean(self._recent_replayed)))
        if self._recent_fresh:
            self.logger.record("replay/success_fresh", float(np.mean(self._recent_fresh)))
        self._last_sync = self.num_timesteps


class GearCurriculumCallback(BaseCallback):
    """Strong-body-first curriculum (after Tao et al. 2022, "Learning to Get Up"): every episode draws
    a gear multiplier from [1, gear_max] for the limb actuators, so nominal-strength (x1) episodes are
    always in the mix and the policy can't specialise to brute force. gear_max starts high and steps
    down by `step` whenever the success rate of the UPPER half of the gear range (calm episodes only
    -- harsh ones are ~0% and would block the schedule forever) reaches `gate`. Evaluate with the
    gate snippet only -- it always uses the nominal body."""

    def __init__(self, start_max=2.0, step=0.05, gate=0.5, sync_every=50_000, window=200, min_n=100):
        super().__init__()
        self.cur_max, self.step, self.gate = start_max, step, gate
        self.sync_every, self.window, self.min_n = sync_every, window, min_n
        self._upper = deque(maxlen=window)
        self._nominal = deque(maxlen=window)
        self._streak = None
        self._last_sync = 0

    def _on_training_start(self):
        self._streak = np.zeros(self.training_env.num_envs)
        self._last_sync = self.num_timesteps
        self.training_env.env_method("set_gear_scale_max", self.cur_max)

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            self._streak[i] = max(self._streak[i], info.get("stand_streak_steps", 0))
            if not self.locals["dones"][i]:
                continue
            ok = 1.0 if self._streak[i] >= STAND_STREAK_REF else 0.0
            self._streak[i] = 0.0
            start = info.get("episode_start")
            if start is None or not start["calm"]:
                continue
            g = start.get("gear", 1.0)
            if g >= (1.0 + self.cur_max) / 2.0 and self.cur_max > 1.0:
                self._upper.append(ok)
            if g <= 1.1:
                self._nominal.append(ok)
        if self.num_timesteps - self._last_sync >= self.sync_every:
            self._sync()
        return True

    def _sync(self):
        if self.cur_max > 1.0 and len(self._upper) >= self.min_n and np.mean(self._upper) >= self.gate:
            self.cur_max = max(1.0, round(self.cur_max - self.step, 3))
            self._upper.clear()
            self.training_env.env_method("set_gear_scale_max", self.cur_max)
        self.logger.record("gear/max", self.cur_max)
        if self._upper:
            self.logger.record("gear/success_upper_half", float(np.mean(self._upper)))
        if self._nominal:
            self.logger.record("gear/success_nominal", float(np.mean(self._nominal)))
        self._last_sync = self.num_timesteps


class ColoredNoiseCallback(BaseCallback):
    """Temporally CORRELATED exploration noise, no policy change (checkpoints stay loadable).

    PPO's default rollout action is mean + std * eps_t with eps_t independent every step; on a body that
    low-passes its torques that averages out, so a ~10-step coordinated push never happens (measured:
    scaling white noise x1..x3 left the share of floor starts whose head ever passes 0.5 m at 4-6%, while
    correlated noise rose to 10%). Here eps_t follows a stationary AR(1) with correlation `rho`, so each
    step's marginal is still N(0,1) and the stored log-prob is the policy's own. On envs currently on the
    floor (torso z < `floor_z`) the noise is additionally multiplied by `floor_scale`; that part makes the
    sampling distribution wider than the policy's (a small off-policy bias that PPO's ratio clipping
    tolerates) and is deliberately limited to the floor so calm balance training is untouched.
    Only rollout sampling is replaced -- policy.predict (evaluation, the gates) is unaffected."""

    def __init__(self, rho=0.9, floor_scale=1.5, floor_z=0.5, burst_rate=0.0, burst_hold=10):
        super().__init__()
        self.rho, self.floor_scale, self.floor_z = rho, floor_scale, floor_z
        # floor bursts: on envs lying on the floor, with per-step probability `burst_rate` (when idle) the
        # limb actions are replaced by a random +-1 pattern held for `burst_hold` steps. Policy-free
        # bang-bang holds of ~10 steps got the head above 0.5 m in 77% of floor starts (the policy: 6%).
        self.burst_rate, self.burst_hold = burst_rate, burst_hold
        self._orig_forward = None
        self.low = None

    def _on_training_start(self):
        import torch as th
        pol = self.model.policy
        n, a_dim = self.training_env.num_envs, int(np.prod(pol.action_space.shape))
        self.eps = th.randn(n, a_dim)
        self.low = th.zeros(n, dtype=th.bool)
        self.burst_left = th.zeros(n, dtype=th.long)
        self.burst_act = th.zeros(n, a_dim)
        self.burst_active_frac = 0.0
        self.override_mask = None
        self._orig_forward = pol.forward
        venv = self.training_env
        self._orig_step_async = venv.step_async
        outer0 = self

        def step_async(actions):
            m = outer0.override_mask
            if m is not None and bool(m.any()):
                actions = np.array(actions, copy=True)
                mm = m.numpy()
                actions[mm] = np.clip(outer0.burst_act.numpy()[mm], -1.0, 1.0)
            outer0.override_mask = None
            return outer0._orig_step_async(actions)

        venv.step_async = step_async
        rho, sc = self.rho, self.floor_scale
        outer = self

        def forward(obs, deterministic=False):
            if deterministic or obs.shape[0] != n:
                return outer._orig_forward(obs, deterministic)
            features = pol.extract_features(obs)
            if pol.share_features_extractor:
                latent_pi, latent_vf = pol.mlp_extractor(features)
            else:
                pi_f, vf_f = features
                latent_pi, latent_vf = pol.mlp_extractor.forward_actor(pi_f), pol.mlp_extractor.forward_critic(vf_f)
            values = pol.value_net(latent_vf)
            dist = pol._get_action_dist_from_latent(latent_pi)
            outer.eps = rho * outer.eps + (1.0 - rho ** 2) ** 0.5 * th.randn_like(outer.eps)
            scale = th.where(outer.low, th.full((n,), sc), th.ones(n)).unsqueeze(1).to(obs.device)
            actions = dist.distribution.mean + scale * dist.distribution.stddev * outer.eps.to(obs.device)
            if outer.burst_rate > 0.0:
                idle = outer.low & (outer.burst_left == 0)
                start = idle & (th.rand(n) < outer.burst_rate)
                if start.any():
                    new = th.sign(th.randn(n, a_dim))
                    new[:, 0] = 0.0                        # action 0 (root_x) is zeroed in stand-only anyway
                    outer.burst_act[start] = new[start]
                    outer.burst_left[start] = outer.burst_hold
                active = outer.low & (outer.burst_left > 0)
                # NOT stored as the policy's action: a +-1 pattern is far out in the policy's tails and blew up
                # PPO's importance ratios (approx_kl ~1e6 -> NaN, first v33 attempt). The policy keeps its own
                # on-policy sample; the override is applied to the env in step_async below, i.e. the burst acts as
                # an environment disturbance that creates raised-head states for the policy to continue from.
                outer.override_mask = active.clone()
                outer.burst_left = th.where(active, outer.burst_left - 1, th.zeros_like(outer.burst_left))
                outer.burst_active_frac = float(active.float().mean())
            log_prob = dist.log_prob(actions)
            return actions.reshape((-1, *pol.action_space.shape)), values, log_prob

        pol.forward = forward

    def _on_step(self):
        import torch as th
        infos, dones = self.locals["infos"], self.locals["dones"]
        for i, info in enumerate(infos):
            self.low[i] = bool(info.get("torso_z", 1.0) < self.floor_z)
            if dones[i]:
                self.eps[i] = th.randn(self.eps.shape[1])     # new episode: fresh stationary draw
                self.burst_left[i] = 0
        return True

    def _on_rollout_end(self):
        self.logger.record("noise/floor_env_frac", float(self.low.float().mean()))
        self.logger.record("noise/burst_active_frac", self.burst_active_frac)

    def _on_training_end(self):
        if self._orig_forward is not None:
            del self.model.policy.forward              # drop the instance override -> class forward again
            if '_orig_step_async' in self.__dict__ and 'step_async' in self.training_env.__dict__:
                del self.training_env.step_async       # back to the VecEnv's own step_async


class BurstArchiveCallback(BaseCallback):
    """Archive of high-head states reached from the floor, replayed as episode starts (pure RL: a reset
    distribution, no off-policy actions). The env reports states with the head 0.25-1.10 m, a foot down and the
    torso not inverted, that are at least ARCHIVE_GAIN above where THAT episode started (info["archive_state"]) --
    from floor starts AND from archive-started episodes, so each rung seeds the next one (Go-Explore-style frontier);
    they are kept in head-height bins so no bin starves, and every `sync_every` steps handed back to the envs via set_replay_pool: a fraction
    `frac` of ALL resets then loads one of them, so the policy practises 'continue from sitting / kneeling' where the
    body can demonstrably get to from a lying start. Logs archive/size and the share of archive-started episodes that
    hold a 3 s stand (archive/success_replayed)."""

    BINS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.90, 1.10)   # lying -> just stood up (7 rungs)

    def __init__(self, frac=0.25, cap_per_bin=1500, sync_every=50_000, min_pool=200):
        super().__init__()
        self.frac, self.cap, self.sync_every, self.min_pool = frac, cap_per_bin, sync_every, min_pool
        self.bins = [deque(maxlen=cap_per_bin) for _ in range(len(self.BINS) - 1)]
        self._streak = None
        self._last_sync = 0
        self._recent = deque(maxlen=300)

    def _on_training_start(self):
        self._streak = np.zeros(self.training_env.num_envs)
        self._last_sync = self.num_timesteps
        self.training_env.env_method("set_archive_collect", True)
        self.training_env.env_method("set_replay_pool", [], self.frac)

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            self._streak[i] = max(self._streak[i], info.get("stand_streak_steps", 0))
            st = info.get("archive_state")
            if st is not None:
                q, v, hz = st
                b = int(np.digitize(hz, self.BINS)) - 1
                if 0 <= b < len(self.bins):
                    self.bins[b].append((q, v, False))
            if self.locals["dones"][i]:
                start = info.get("episode_start")
                if start is not None and start.get("replayed"):
                    self._recent.append(1.0 if self._streak[i] >= STAND_STREAK_REF else 0.0)
                self._streak[i] = 0.0
        if self.num_timesteps - self._last_sync >= self.sync_every:
            self._sync()
        return True

    def _sync(self):
        sizes = [len(b) for b in self.bins]
        if sum(sizes) >= self.min_pool:
            # equal weight per head-height bin (low poses are far more common; sampling with replacement
            # lets the rarer, higher bins count as much as the common low one)
            rng = np.random.default_rng(self.num_timesteps)
            target = min(max(sizes), 600)
            pool = []
            for b in self.bins:
                if len(b):
                    idx = rng.choice(len(b), size=target, replace=len(b) < target)
                    pool.extend(b[j] for j in idx)
            self.training_env.env_method("set_replay_pool", pool, self.frac)
        self.logger.record("archive/size", sum(sizes))
        for k, n in enumerate(sizes):
            self.logger.record(f"archive/bin{k}_{self.BINS[k]:.2f}m", n)
        if self._recent:
            self.logger.record("archive/success_replayed", float(np.mean(self._recent)))
        self._last_sync = self.num_timesteps


class HandoffPoolCallback(BaseCallback):
    """Starts a fraction of episodes from states the FLOOR get-up policy hands over (make_handoff_pool.py / make_strict_pool.py).

    The stand policy never practised those states (it caught ~10% of them), so they are handed to the envs as the replay pool
    (set_replay_pool, same mechanism as the failed-start replay); the rest of the resets keep the normal calm / harsh distribution.
    The pool is fixed (not refreshed from the policy being trained).

    Three modes for the pool share:
      constant        (default)               `frac` from step 0 -- v39 with 0.4 ended up falling flat even from upright starts
      step ramp       ramp_steps > 0          0 before ramp_start, then linear up to `frac` over ramp_steps
      performance gate gate_calm is not None  every gate_every steps a deterministic calm gate (gate_n fixed seeds, 3 s, the same test as
                                              stand_progress_notify) runs inside the trainer: share += gate_step if the gate fraction is
                                              >= gate_calm, share -= gate_step if it is < gate_calm / 2, else unchanged; capped at `frac`
                                              (same idea as CurriculumGateCallback: advance on results, not on a timer)"""

    UPDATE_EVERY = 100_000

    def __init__(self, path, frac=0.4, ramp_start=0, ramp_steps=0, gate_calm=None, gate_step=0.025, gate_every=500_000, gate_n=40,
                 fail_weight=0, fail_every=1_000_000, fail_n=300):
        super().__init__()
        self.path, self.frac = path, frac
        self.ramp_start, self.ramp_steps = ramp_start, ramp_steps
        self.gate_calm, self.gate_step, self.gate_every, self.gate_n = gate_calm, gate_step, gate_every, gate_n
        self._pool, self._last_update, self._cur = None, -10**12, None
        self._gated, self._last_gate, self._gate_env, self._gate_val = 0.0, 0, None, None
        self.fail_weight, self.fail_every, self.fail_n = fail_weight, fail_every, fail_n
        self._status, self._fail_cursor, self._last_fail_eval, self._pool_env = None, 0, 0, None

    def _frac_now(self):
        if self.gate_calm is not None:
            return self._gated
        if self.ramp_steps <= 0:
            return self.frac
        t = (self.num_timesteps - self.ramp_start) / self.ramp_steps
        return self.frac * min(1.0, max(0.0, t))

    def _push(self):
        cur = round(self._frac_now(), 4)
        if cur != self._cur:
            self.training_env.env_method("set_replay_pool", self._pool, cur)
            self._cur = cur
        self._last_update = self.num_timesteps

    def _calm_gate(self):
        from env import FALL_HEIGHT
        from locomotion_env import LocomotionEnv
        if self._gate_env is None:
            self._gate_env = LocomotionEnv(stand_only=True)
            self._gate_env.set_calm_frac(1.0)
        env, ok = self._gate_env, 0
        for seed in range(self.gate_n):
            obs, _ = env.reset(seed=seed)
            fell = False
            for _ in range(150):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _r, term, trunc, _info = env.step(action)
                if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT:
                    fell = True
                    break
                if term or trunc:
                    break
            ok += (not fell)
        return ok / self.gate_n

    def _hold_from_pool_state(self, i):
        """3 s hold test from pool state i with the current policy (deterministic): True if the torso never drops below FALL_HEIGHT."""
        import mujoco
        from env import FALL_HEIGHT
        from locomotion_env import LocomotionEnv
        if self._pool_env is None:
            self._pool_env = LocomotionEnv(stand_only=True)
        env = self._pool_env
        env.reset(seed=i)
        q, v, _ = self._pool[i]
        env.data.qpos[:], env.data.qvel[:] = q, v
        mujoco.mj_forward(env.model, env.data)
        env._prev_com_x = None
        obs = env._obs()
        for _ in range(150):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _info = env.step(action)
            if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT:
                return False
            if term or trunc:
                break
        return True

    def _fail_weighted_refresh(self):
        """Re-test a rotating chunk of the pool with the current policy, then rebuild the replay list: states it fails are repeated
        `fail_weight` times, passed ones once, never-tested ones twice (so the policy keeps practising what it cannot yet catch)."""
        n = len(self._pool)
        idx = [(self._fail_cursor + k) % n for k in range(min(self.fail_n, n))]
        self._fail_cursor = (self._fail_cursor + len(idx)) % n
        passed = 0
        for i in idx:
            ok = self._hold_from_pool_state(i)
            self._status[i] = 1 if ok else 2
            passed += ok
        weights = np.where(self._status == 2, self.fail_weight, np.where(self._status == 1, 1, 2))
        expanded = [self._pool[i] for i in range(n) for _ in range(int(weights[i]))]
        self.training_env.env_method("set_replay_pool", expanded, round(self._frac_now(), 4))
        self._cur = round(self._frac_now(), 4)
        known = int((self._status > 0).sum()); failed = int((self._status == 2).sum())
        self._pass_rate = passed / max(len(idx), 1)
        print(f"[locomotion] pool refresh at {self.num_timesteps} steps: chunk pass {passed}/{len(idx)}, known {known}/{n}, failing {failed}, "
              f"replay list {len(expanded)}", flush=True)

    def _on_training_start(self):
        d = np.load(self.path, allow_pickle=True)
        self._pool = [(q, v, False) for q, v in zip(d["qpos"], d["qvel"])]
        self._status = np.zeros(len(self._pool), dtype=np.int8)
        self._last_gate = self.num_timesteps
        self._last_fail_eval = self.num_timesteps
        self._pass_rate = None
        self._push()
        mode = (f", performance gate: calm gate >= {self.gate_calm} -> +{self.gate_step}, < {self.gate_calm / 2} -> -{self.gate_step}, every {self.gate_every} steps, max {self.frac}"
                if self.gate_calm is not None else
                f", ramp 0 -> {self.frac} over {self.ramp_steps} steps from step {self.ramp_start}" if self.ramp_steps > 0 else "")
        print(f"[locomotion] hand-off pool: {len(self._pool)} states from {self.path} ({d['floor_ckpt']}), frac {self._cur}{mode}", flush=True)

    def _on_step(self):
        if self.fail_weight > 0 and self.num_timesteps - self._last_fail_eval >= self.fail_every:
            self._last_fail_eval = self.num_timesteps
            self._fail_weighted_refresh()
            self.logger.record("handoff/pool_chunk_pass", self._pass_rate)
        if self.gate_calm is not None:
            if self.num_timesteps - self._last_gate >= self.gate_every:
                self._last_gate = self.num_timesteps
                g = self._calm_gate()
                self._gate_val = g
                if g >= self.gate_calm:
                    self._gated = min(self.frac, self._gated + self.gate_step)
                elif g < self.gate_calm / 2:
                    self._gated = max(0.0, self._gated - self.gate_step)
                self._push()
                print(f"[locomotion] pool gate at {self.num_timesteps} steps: calm gate {g:.3f} -> pool share {self._cur}", flush=True)
            self.logger.record("handoff/frac", self._cur)
            if self._gate_val is not None:
                self.logger.record("handoff/calm_gate_inproc", self._gate_val)
        elif self.ramp_steps > 0:
            if self.num_timesteps - self._last_update >= self.UPDATE_EVERY:
                self._push()
            self.logger.record("handoff/frac", self._cur)
        return True


class FallRateCallback(BaseCallback):
    """Fraction of completed episodes that ended via a_out (failed to recover within
    DOWN_RECOVERY_STEPS) rather than truncation at MAX_STEPS -- the direct metric this
    experiment is about, logged live instead of only via a separate eval pass."""

    def __init__(self, window=200):
        super().__init__()
        self._recent = deque(maxlen=window)

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            if self.locals["dones"][i]:
                self._recent.append(1.0 if info.get("a_out") else 0.0)
        if self._recent:
            self.logger.record("rollout/fall_rate", float(np.mean(self._recent)))
        return True


def main():
    global LOG_STD_MIN, LOG_STD_MAX
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="ppo_locomotion")
    parser.add_argument("--init-from", type=str, default=None,
                         help="checkpoint to warm-start from (any Fighter2DEnv-compatible "
                              "policy, e.g. a combat checkpoint) -- default: train from scratch")
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--knee-already-relaxed", action="store_true",
                         help="skip the knee scaffold straight to its permanent floor from "
                              "step 0 (docs 10.44/10.45) -- pass this on any --init-from "
                              "continuation of a run whose knee schedule already finished")
    parser.add_argument("--stand-only", action="store_true",
                         help="zero the velocity and gait rewards -- learn standing from any posture alone")
    parser.add_argument("--assist-already-relaxed", action="store_true",
                         help="same as --knee-already-relaxed, for the velocity assist scaffold")
    parser.add_argument("--calm-already-relaxed", action="store_true",
                         help="same as --knee-already-relaxed, for the stand-only calm-episode-"
                              "fraction curriculum (only meaningful with --stand-only)")
    parser.add_argument("--ground-shaping", action="store_true",
                         help="enable LocomotionEnv ground_shaping (com gated by recovery_gate + "
                              "linear low-height reward) -- only meaningful with --stand-only")
    parser.add_argument("--getup-reward", action="store_true",
                         help="LocomotionEnv getup_reward: implies --ground-shaping, plus foot-contact-"
                              "gated height, weight-shift term, head-contact/inversion penalties, and "
                              "upright-only stand_streak -- only meaningful with --stand-only")
    parser.add_argument("--head-progress", type=float, default=0.0,
                         help="K for the potential-based head-height progress reward (pays for RISING; "
                              "0 = off). Head height is capped at 0.9 m; stand-only only")
    parser.add_argument("--head-record", type=float, default=0.0,
                         help="K for the head-height RECORD bonus: pays K * (new best head height - previous "
                              "best) once per height per episode, only with a foot on the floor (0 = off; "
                              "stand-only). Not cancelled by falling back, not farmable by bobbing")
    parser.add_argument("--colored-noise-rho", type=float, default=0.0,
                         help="ColoredNoiseCallback: AR(1) correlation of the rollout exploration noise "
                              "(0 = off = SB3's independent-per-step noise)")
    parser.add_argument("--floor-noise-scale", type=float, default=1.5,
                         help="with --colored-noise-rho: extra multiplier on the noise for envs whose torso is "
                              "below 0.5 m (1.0 = none)")
    parser.add_argument("--floor-burst-rate", type=float, default=0.0,
                         help="with --colored-noise-rho: per-step probability (when idle) that an env lying on "
                              "the floor switches to a random +-1 limb pattern held --floor-burst-hold steps "
                              "(0 = off). ~0.1 puts about half of the floor time in bursts")
    parser.add_argument("--floor-burst-hold", type=int, default=10)
    parser.add_argument("--burst-archive-frac", type=float, default=0.0,
                         help="BurstArchiveCallback: fraction of resets that start from archived high-head "
                              "states reached from the floor (0 = off; stand-only)")
    parser.add_argument("--handoff-pool", type=str, default=None,
                         help="npz from make_handoff_pool.py: start --handoff-frac of the episodes from the states the "
                              "floor get-up policy hands over (stand-only)")
    parser.add_argument("--handoff-frac", type=float, default=0.4,
                         help="share of episode starts drawn from the pool (the END value when --handoff-ramp-steps > 0)")
    parser.add_argument("--handoff-gate-calm", type=float, default=None,
                         help="performance-gated pool share: every --handoff-gate-every steps run a deterministic calm gate inside the trainer; "
                              "share += --handoff-gate-step if the gate fraction >= this, -= if < half of it (cap: --handoff-frac)")
    parser.add_argument("--handoff-fail-weight", type=int, default=0,
                         help="> 0: every --handoff-fail-every steps re-test a chunk of the pool with the current policy and repeat the states it fails "
                              "this many times in the replay list (failure-weighted replay); 0 = uniform")
    parser.add_argument("--handoff-fail-every", type=int, default=1_000_000)
    parser.add_argument("--handoff-gate-step", type=float, default=0.025)
    parser.add_argument("--handoff-gate-every", type=int, default=500_000)
    parser.add_argument("--handoff-ramp-start", type=int, default=0, help="step at which the pool share starts to rise (with --handoff-ramp-steps)")
    parser.add_argument("--handoff-ramp-steps", type=int, default=0,
                         help="> 0: raise the pool share linearly from 0 to --handoff-frac over this many steps (default 0 = constant share)")
    parser.add_argument("--std-min", type=float, default=None,
                         help="raise the lower bound of the policy action std (default keeps 0.05); also applied "
                              "once at load so a shrunken std is lifted immediately")
    parser.add_argument("--std-max", type=float, default=None,
                         help="upper bound of the policy action std (default 0.6, which is also the starting std because the clamp pulls the "
                              "initial std down to it). The action is multiplied by the actuator gear, so the same std is a 1.5x larger torque noise on "
                              "the x1.5 body: 0.4 keeps the torque noise of the x1 runs")
    parser.add_argument("--knee-gate", action="store_true",
                         help="fade the knee_avoid reward out by head height (0 below 0.5 m, full at 0.9 m) so "
                              "raising the legs while lying isn't paid -- only meaningful with --stand-only")
    parser.add_argument("--replay-curriculum", action="store_true",
                         help="ReplayCurriculumCallback: replay failed start states from buckets of "
                              "intermediate difficulty (only meaningful with --stand-only)")
    parser.add_argument("--gear-curriculum", action="store_true",
                         help="GearCurriculumCallback: per-episode gear multiplier in [1, gear_max], gear_max "
                              "stepped down by success (only meaningful with --stand-only)")
    parser.add_argument("--gear-start", type=float, default=2.0,
                         help="initial gear_max for --gear-curriculum (use the logged gear/max when resuming)")
    parser.add_argument("--replay-frac", type=float, default=0.3,
                         help="fraction of resets drawn from the failed-start pool")
    parser.add_argument("--calm-end-frac", type=float, default=None,
                         help="CurriculumGateCallback phase-2 target calm fraction -- default is "
                              "locomotion_env.CALM_EPISODE_FRAC; lower = more harsh/collapse starts")
    parser.add_argument("--height-already-relaxed", action="store_true",
                         help="same as --knee-already-relaxed, for the stand-only head-height "
                              "reward scaffold (only meaningful with --stand-only)")
    parser.add_argument("--height-start-scale", type=float, default=6.0,
                         help="HeightRewardScaffoldCallback start value -- override when "
                              "continuing a run that's already past the original 6.0 baseline")
    parser.add_argument("--height-end-scale", type=float, default=10.0,
                         help="HeightRewardScaffoldCallback target value")
    parser.add_argument("--height-relax-frac", type=float, default=0.6,
                         help="HeightRewardScaffoldCallback ramp fraction of --timesteps")
    parser.add_argument("--curriculum-fall-gate", type=float, default=None,
                         help="CurriculumGateCallback.FALL_RATE_GATE override -- loosen when "
                              "phase 1 has plateaued below the strict 0.1 default and it's time "
                              "to start mixing in harsh/collapse starts anyway")
    parser.add_argument("--curriculum-streak-gate", type=float, default=None,
                         help="CurriculumGateCallback.STREAK_GATE_STEPS override (pairs with "
                              "--curriculum-fall-gate)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_dir = MODELS_DIR / "autosave"
    ckpt_dir.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.out}_{run_id}"

    vec_env = make_vec_env(LocomotionEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
                           env_kwargs={"stand_only": args.stand_only,
                                       "ground_shaping": args.ground_shaping,
                                       "getup_reward": args.getup_reward,
                                       "knee_gate": args.knee_gate,
                                       "head_progress": args.head_progress,
                                       "head_record": args.head_record})

    if args.init_from:
        custom_objects = {"ent_coef": args.ent_coef} if args.ent_coef is not None else None
        model = PPO.load(args.init_from, env=vec_env, device=args.device, custom_objects=custom_objects)
        print(f"[locomotion] warm-starting from {args.init_from}", flush=True)
    else:
        kwargs = {}
        if args.ent_coef is not None:
            kwargs["ent_coef"] = args.ent_coef
        model = PPO("MlpPolicy", vec_env, device=args.device, verbose=1, **kwargs)
        print("[locomotion] training from scratch", flush=True)

    if args.std_max is not None:
        LOG_STD_MAX = float(np.log(args.std_max))
        with torch.no_grad():
            before_std = model.policy.log_std.exp().mean().item()
            model.policy.log_std.data.clamp_(min=LOG_STD_MIN, max=LOG_STD_MAX)
            print(f"[locomotion] std cap {args.std_max}: mean action std {before_std:.3f} -> {model.policy.log_std.exp().mean().item():.3f}", flush=True)
    if args.std_min is not None:
        LOG_STD_MIN = float(np.log(args.std_min))
        with torch.no_grad():
            before_std = model.policy.log_std.exp().mean().item()
            model.policy.log_std.data.clamp_(min=LOG_STD_MIN, max=LOG_STD_MAX)
            print(f"[locomotion] std floor {args.std_min}: mean action std {before_std:.3f} -> "
                  f"{model.policy.log_std.exp().mean().item():.3f}", flush=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=run_name,
    )
    callback_list = [checkpoint_callback, BreakdownCallback(), FallRateCallback(),
                      LogStdClampCallback(),
                      KneeScaffoldCallback(args.timesteps, already_relaxed=args.knee_already_relaxed),
                      VelocityAssistCallback(args.timesteps, already_relaxed=args.assist_already_relaxed)]
    if args.stand_only:
        curriculum_kwargs = {} if args.calm_end_frac is None else {"end_frac": args.calm_end_frac}
        callback_list.append(
            CurriculumGateCallback(args.timesteps, already_relaxed=args.calm_already_relaxed,
                                    fall_rate_gate=args.curriculum_fall_gate,
                                    streak_gate_steps=args.curriculum_streak_gate,
                                    **curriculum_kwargs))
        callback_list.append(
            HeightRewardScaffoldCallback(args.timesteps, start_scale=args.height_start_scale,
                                          end_scale=args.height_end_scale,
                                          relax_frac=args.height_relax_frac,
                                          already_relaxed=args.height_already_relaxed))
    if args.stand_only and args.replay_curriculum:
        callback_list.append(ReplayCurriculumCallback(frac=args.replay_frac))
    if args.stand_only and args.handoff_pool:
        callback_list.append(HandoffPoolCallback(args.handoff_pool, frac=args.handoff_frac,
                                                 ramp_start=args.handoff_ramp_start, ramp_steps=args.handoff_ramp_steps,
                                                 gate_calm=args.handoff_gate_calm, gate_step=args.handoff_gate_step,
                                                 gate_every=args.handoff_gate_every,
                                                 fail_weight=args.handoff_fail_weight, fail_every=args.handoff_fail_every))
    if args.stand_only and args.burst_archive_frac > 0.0:
        callback_list.append(BurstArchiveCallback(frac=args.burst_archive_frac))
    if args.colored_noise_rho > 0.0:
        callback_list.append(ColoredNoiseCallback(rho=args.colored_noise_rho, floor_scale=args.floor_noise_scale,
                                                    burst_rate=args.floor_burst_rate, burst_hold=args.floor_burst_hold))
    if args.stand_only and args.gear_curriculum:
        callback_list.append(GearCurriculumCallback(start_max=args.gear_start))
    callbacks = CallbackList(callback_list)

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=(args.init_from is None))
    finally:
        with torch.no_grad():
            before = model.policy.log_std.data.clone()
            model.policy.log_std.data.clamp_(min=LOG_STD_MIN, max=LOG_STD_MAX)
            n_clamped = int((model.policy.log_std.data != before).sum())
        if n_clamped:
            print(f"[locomotion] clamped log_std on {n_clamped}/{before.numel()} action dims "
                  f"into [{np.exp(LOG_STD_MIN):.3f}, {np.exp(LOG_STD_MAX):.3f}] std range", flush=True)
        out_path = MODELS_DIR / run_name
        model.save(str(out_path))
        print(f"saved model to {out_path}.zip")


if __name__ == "__main__":
    main()
