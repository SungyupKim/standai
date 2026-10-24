"""Floor-only get-up environment: a separate, small problem from "stand and balance".

Every episode starts with the body lying/collapsed (torso < FLOOR_START_Z). The task is just to get the
head up to HANDOFF_HEAD_Z with a foot on the ground; reaching it ends the episode as a success and the
reached state is reported in info["handoff_state"] (qpos, qvel, head_z) so the stand policy can practise
from exactly the states this policy hands over. Reward is deliberately plain (no balance / stance terms):

    head_record  K * (new best normalised head height - previous best), foot on the floor only
    floor_dense  small per-step terms for head height / torso height with a foot down
    success      one-off bonus once the hand-off pose has been held for HOLD_STEPS consecutive steps (v4+: 1 = reach)

Hand-off pose = torso >= HOLD_TORSO_Z, head >= HANDOFF_HEAD_Z, a foot down, |pitch| < HANDOFF_MAX_PITCH_DEG.
(v1/v2 ended the episode the first instant head >= 0.9 m with a foot down; watching successes showed every one
of them dropping back to torso 0.3-0.5 m within 3 s, i.e. a pop-up, not a stand.)
"""
import os

import numpy as np

import env as _base_env
from locomotion_env import LocomotionEnv

FLOOR_START_Z = 0.5            # resample the start until the torso is this low (lying / collapsed)
FLOOR_MAX_STEPS = 250          # 5 s, same as the env's own DOWN_RECOVERY_STEPS
HANDOFF_HEAD_Z = 0.9           # head height (m) at which the stand policy takes over
HANDOFF_MAX_PITCH_DEG = 60.0
HOLD_TORSO_Z = 0.6
HOLD_STEPS = 1                 # reach = success (v3 used 20 steps; holding the pose is the STAND policy's job)
# "settled" hand-off (OPT-IN: FloorEnv(settled=True) / train_floor.py --settled; the default stays the v1-v4 condition):
# the loose condition fires while the body is still being thrown up (torso vz ~ +1.5 m/s, pitch rate ~ 100 deg/s) and
# the stand policy v37@96M catches only ~10% of those states; with |vz| < 0.5 and |pitch rate| < 100 deg/s (a single step)
# it caught 24% -- but v4 reaches that state in only 46/200 starts, and NOT for 3 consecutive steps (2/100): those are
# mostly passes through the apex of the throw, not a settled pose. Untested as a training target (v5 was not run).
SETTLED_TORSO_Z = 0.7
SETTLED_MAX_PITCH_DEG = 40.0
SETTLED_MAX_VZ = 0.5           # m/s
SETTLED_MAX_PITCH_RATE = 100.0  # deg/s
SETTLED_HOLD_STEPS = 1         # 3 would leave v4 with 2/100 starts that ever qualify; 1 matches the 46/200 measurement
SETTLE_SCALE = 0.03            # per-step shaping near the top: slower is better (bounded, success pays more)
# "stand-ready" hand-off (OPT-IN: FloorEnv(standready=True) / train_floor.py --standready, floor v6): the floor policy
# keeps going until the body is in the region where the STAND policy is known to hold (upright calm starts: 92%; with
# extra torso vz up to +-1 m/s and pitch rate up to +-100 deg/s it still holds 53-100%, scripts/analysis/stand_velocity_map.py),
# instead of handing over a deep squat (knee ~ -92 deg) that is still rising (stand policy catches ~7%). Starts: lying
# (as before) plus `crouch_frac` crouched calm starts, so the same policy also covers "get up from a crouch".
SR_TORSO_Z = 1.08
SR_MAX_KNEE_BEND_DEG = 20.0     # both knees straighter than -20 deg
SR_MAX_PITCH_DEG = 20.0
SR_MAX_VZ = 1.0                # m/s
SR_MAX_PITCH_RATE = 100.0      # deg/s
# Staged stand-ready target (FloorEnv(level=...) / train_floor.py --level-start): level 1 = the strict thresholds above, level 0 = the
# LOOSE ones below, linear in between; set_level() lets a callback tighten it while training. Foot-down is required at every level.
# Why: prone floor policies (prone1-3, 40M steps) reached torso ~0.93 m (median) but the strict region (1.08 m, knees > -20, pitch < 20,
# slow) was met 0-1 times in 100, so there was no success bonus to learn from.
SR_LOOSE = dict(torso=0.75, knee_bend=110.0, pitch=45.0, vz=2.5, pr=400.0, com=0.30)   # torso was 0.85 in prone4/5/6 (warm starts); 0.75 for a from-scratch run
# Feet-gap condition (OPT-IN: STANDAI_FEET_GAP=<strict gap in m, e.g. 0.15> in the environment of every process that builds the FloorEnv: trainer,
# benchmark, pool builder, notifier). Without it the region only needs ONE foot down, and both v12 and the 2026-10-18 re-trained supine policy hand over
# standing on one leg with the other leg raised to hip height (hip 120 deg, knee straight, feet 1.2 m apart in height); the stand policy then catches 73% of v12's
# hand-offs but only 8% when left and right are swapped, and 4% of the new policy's (mirror image) -- docs 3.22. With the condition the region needs |z_foot_r - z_foot_l|
# <= gap, linear in the level from SR_FEET_GAP_LOOSE (level 0, effectively off) to the strict value (level 1); the dense stand-ready term is also scaled by exp(-(gap/0.5)^2).
SR_FEET_GAP_LOOSE = 1.6
# Stand-policy value bonus (OPT-IN: STANDAI_STAND_VALUE_CKPT=<stand policy zip> + STANDAI_STAND_VALUE_COEF=<k>, standready only): while the foot is down and the torso is at
# least 0.9 m, the floor policy gets an extra k * clip((V_stand(obs) - LO) / (HI - LO), 0, 1) per step, V_stand being the critic of the stand policy that will take over
# (LO / HI = STANDAI_STAND_VALUE_LO / _HI, default 500 / 2500: on prone hand-offs V is 926 for the states v54 misses and 1972 for the ones it catches, AUC 0.95 -- docs 3.22).
# It is only an auxiliary term: the head record / height terms and the hand-off bonus stay the main reward. Process-wide model, so set it only for the trainer, not the notifier.
SR_BONUS = 20.0
SR_DENSE_SCALE = 0.05          # per-step: torso height x knee extension
SR_SETTLE_SCALE = 0.03
SR_MAX_STEPS = 300             # 6 s (lying -> 0.9 m head takes ~1.8 s median; the knee extension comes after)
CROUCH_KNEE_DEG = -25.0        # a calm start counts as "crouched" if a knee is bent more than this ...
CROUCH_TORSO_Z = 1.05          # ... or the torso is lower than this
# v10 opt-in penalties (FloorEnv(speed_pen=..., air_pen=...) / train_floor.py --speed-pen --air-pen), standready only:
#   speed_pen: height x speed. weight h_w = 0 at torso 0.6 m rising to 1 at 1.08 m, speed = |vz|/1.0 + |pitch rate|/100 deg/s
#              (the two tolerances of the stand-ready region). Fast motion is cheap near the floor (rising needs it) and
#              expensive up high, where the stand policy loses balance (pitch rate >~ 100-200 deg/s => 0-43% hold).
#   air_pen:   per step with torso > 0.6 m and NO foot on the floor (hopping / leaping up instead of standing up).
SP_H_LO = 0.6
SP_H_HI = 1.08
RECORD_K = 10.0
DENSE_SCALE = 0.02
TORSO_DENSE_SCALE = 0.03
SUCCESS_BONUS = 10.0
_MAX_RESAMPLE = 30


class FloorEnv(LocomotionEnv):
    def __init__(self, settled=False, standready=False, crouch_frac=0.4, speed_pen=0.0, air_pen=0.0, speed_vz_w=1.0, max_steps=None, prone_frac=0.0, down_timeout=None, level=1.0, com_max=None, supine_frac=0.0, **kwargs):
        super().__init__(stand_only=True, **kwargs)
        self._level = float(level)
        # optional stand-ready condition: centre of mass x minus the FRONT foot x <= com_max (m). Hand-off states the stand policy
        # does not hold have the COM ~10 cm ahead of the front foot, held ones ~2 cm (AUC 0.26, 163 prone hand-offs of v4 @41M).
        self._com_max = com_max
        _fg = os.environ.get("STANDAI_FEET_GAP")
        self._feet_gap = float(_fg) if _fg else None
        self._sv_coef = float(os.environ.get("STANDAI_STAND_VALUE_COEF", "0") or 0)
        self._sv_model = None
        self._sv_bonus = float(os.environ.get("STANDAI_STAND_VALUE_BONUS", "10"))   # extra hand-off bonus * V_norm at the success step (success bonus itself is 20)
        if self._sv_coef > 0.0:
            import torch
            from stable_baselines3 import PPO
            torch.set_num_threads(1)
            self._sv_torch = torch
            self._sv_model = PPO.load(os.environ["STANDAI_STAND_VALUE_CKPT"], device="cpu")
            self._sv_sym = os.environ.get("STANDAI_STAND_VALUE_SYM") == "1"      # value of the better of the two left-right copies (scripts/sym.py)
            if self._sv_sym:
                from sym import SymPolicy
                self._sv_symp = SymPolicy(self._sv_model)
            self._sv_lo = float(os.environ.get("STANDAI_STAND_VALUE_LO", "500"))
            self._sv_hi = float(os.environ.get("STANDAI_STAND_VALUE_HI", "2500"))
        self._root_body = int(self.model.body_rootid[self.a_torso_id])
        # env.py ends an episode as a KO once the torso has stayed below FALL_HEIGHT for DOWN_RECOVERY_STEPS (250 = 5 s)
        # in a row; a floor policy that is still on the floor at 5 s therefore never sees the 450-step limit. down_timeout=False
        # (or STANDAI_FLOOR_NO_DOWN_TIMEOUT=1) disables that rule for this process so only max_steps ends a floor episode.
        # Process-wide (it patches env.DOWN_RECOVERY_STEPS), which is fine: floor processes only run FloorEnv.
        if down_timeout is None:
            down_timeout = os.environ.get("STANDAI_FLOOR_NO_DOWN_TIMEOUT") != "1"
        if not down_timeout:
            _base_env.DOWN_RECOVERY_STEPS = 10 ** 9
        # share of lying starts forced to the start-pitch <= 0 side (head toward +x; inferred face-down/prone, not verified).
        # The natural split is ~80/20 and v12 @101.5M reaches 51% of the + side but 1/41 of the - side (200 seeds), so training
        # oversamples it. 0 = natural distribution (the evaluations keep 0, so fixed-seed numbers stay comparable).
        self._prone_frac = prone_frac
        # STANDAI_FLAT_FRAC=<share>: that share of lying starts is forced to FLAT legs (mean hip angle < STANDAI_FLAT_HIP_DEG, default 10 deg). v12's supine starts that reach the
        # hand-off region have the hips bent (median +32 deg), the ones that never get up have them straight (median -2 deg; docs 3.24 addendum 3).
        # STANDAI_ARM_PUSH_COEF=<c>: while the torso is below 0.7 m and the head is rising (> 0.05 m/s), pay c * min(arm contact force / body weight, 1.5) per step. v12 gets up by
        # swinging its arms (arm push 0.14 x weight); the prone policy pushes up with them (0.60 x weight when it succeeds, 0.11 when it fails; docs 3.24 addendum 3).
        self._arm_coef = float(os.environ.get("STANDAI_ARM_PUSH_COEF", "0") or 0)
        self._arm_bodies = {"a_upper_arm_r", "a_forearm_r", "a_upper_arm_l", "a_forearm_l"}
        self._prev_head_z = None
        # STANDAI_ARRIVAL_K=<k> (OPT-IN, standready only): at the success step pay k * (clip(COM forward speed, -0.2, V)/V - clip((pitch_deg - 3)/20, 0, 1)), V = STANDAI_ARRIVAL_VX (default 0.45 m/s).
        # Supine arrivals that end standing carry COM speed +0.44 m/s and ~1 deg pitch, the ones that do not ~0 m/s and ~7.5 deg (docs 3.24 addendum 10): a continuous physical arrival score.
        self._arr_k = float(os.environ.get("STANDAI_ARRIVAL_K", "0") or 0)
        self._arr_v = float(os.environ.get("STANDAI_ARRIVAL_VX", "0.45"))
        self._arr_prev_com_x = None
        # STANDAI_CATCH_K=<k> + STANDAI_CATCH_STAND=<stand policy zip> (OPT-IN, standready only): at the success step the frozen stand policy is run for STANDAI_CATCH_STEPS (default 150) steps from
        # the arrival state and k is paid if it CATCHES it (torso >= 0.7, head >= 0.9, |pitch| < 40 deg for the last 20 steps = benchmark/catch_by_com criterion). The episode ends at success anyway, so
        # the rollout only has to read the outcome (deterministic stand policy -> the reward is a deterministic function of the arrival state). docs 3.24 addendum 11.
        self._catch_k = float(os.environ.get("STANDAI_CATCH_K", "0") or 0)
        self._catch_n = int(os.environ.get("STANDAI_CATCH_STEPS", "150"))
        self._catch_early = int(os.environ.get("STANDAI_CATCH_EARLY", "0") or 0)
        self._catch_model = None
        if self._catch_k > 0.0:
            from sym import load_stand
            import torch; torch.set_num_threads(1)
            self._catch_model = load_stand(os.environ["STANDAI_CATCH_STAND"])
        self._flat_frac = float(os.environ.get("STANDAI_FLAT_FRAC", "0") or 0)
        self._flat_hip = np.deg2rad(float(os.environ.get("STANDAI_FLAT_HIP_DEG", "10")))
        self._hip_adr = [self.model.joint("a_hip_r").qposadr[0], self.model.joint("a_hip_l").qposadr[0]]
        self._supine_frac = supine_frac      # same for the start-pitch > 0 side (inferred supine); prone_frac and supine_frac are exclusive
        self._start_side = "pos"
        # episode length: None = SR_MAX_STEPS (standready) / FLOOR_MAX_STEPS. v11 @98.8M, lying starts, 100 seeds: 300 steps
        # reached 31 / held 16, 450 steps 39 / 20, 750 steps 42 / 22 (the median reach is at step 202, the top quartile >310)
        self._max_steps = max_steps if max_steps is not None else (SR_MAX_STEPS if standready else FLOOR_MAX_STEPS)
        self._settled = settled
        self._standready = standready
        self._crouch_frac = crouch_frac
        self._speed_pen = speed_pen
        self._air_pen = air_pen
        self._speed_vz_w = speed_vz_w     # weight of |vz| inside the speed term (the binding quantity is the pitch rate, not vz)
        self._start_kind = "lying"
        self._ry_dof = self.model.joint("a_root_ry").dofadr[0]
        self._calm_frac = 0.0
        self._floor_head_best = 0.0
        self._floor_head_max = 0.0
        self._hold = 0

    def set_calm_frac(self, frac):          # the callbacks may broadcast this; a floor env is never calm
        self._calm_frac = 0.0

    def _knee_min_deg(self):
        return float(np.degrees(min(self.data.qpos[self._knee_r_qpos], self.data.qpos[self._knee_l_qpos])))

    def set_level(self, level):             # called through env_method by the curriculum callback
        self._level = float(min(1.0, max(0.0, level)))

    def _sr_thresholds(self):
        """(torso_z, knee_bend_deg, pitch_deg, vz, pitch_rate_deg) of the stand-ready region at the current level."""
        a = self._level
        lo = SR_LOOSE
        mix = lambda loose, strict: loose + a * (strict - loose)
        return (mix(lo["torso"], SR_TORSO_Z), mix(lo["knee_bend"], SR_MAX_KNEE_BEND_DEG), mix(lo["pitch"], SR_MAX_PITCH_DEG),
                mix(lo["vz"], SR_MAX_VZ), mix(lo["pr"], SR_MAX_PITCH_RATE))

    def _feet_gap_now(self):
        """|z_foot_r - z_foot_l| (m) and its limit at the current level (None = condition off)."""
        fz = [float(self.data.site_xpos[sid][2]) for sid in self.a_feet]
        gap = abs(fz[0] - fz[1])
        lim = None if self._feet_gap is None else SR_FEET_GAP_LOOSE + self._level * (self._feet_gap - SR_FEET_GAP_LOOSE)
        return gap, lim

    def _com_limit(self):
        """COM-ahead limit at the current level: SR_LOOSE['com'] at level 0 -> com_max at level 1 (None = condition off)."""
        if self._com_max is None:
            return None
        return SR_LOOSE["com"] + self._level * (self._com_max - SR_LOOSE["com"])

    def _com_ahead(self):
        """COM x minus the front foot x (m); positive = the centre of mass is ahead of the support (assumes the body faces +x)."""
        return float(self.data.subtree_com[self._root_body][0] - max(self.data.site_xpos[s][0] for s in self.a_feet))

    def _arm_force_ratio(self):
        """Normal force of the floor on the arms (upper arms + forearms) divided by the body weight."""
        import mujoco
        names = getattr(self, "_geom_body_names", None)
        if names is None:
            self._getup_contacts()
            names = self._geom_body_names
        f6 = np.zeros(6)
        tot = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1, n2 = names[c.geom1], names[c.geom2]
            for n, o in ((n1, n2), (n2, n1)):
                if n in self._arm_bodies and not (o.startswith("a_") or o.startswith("b_")):
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    tot += f6[0]
        return tot / (float(self._a_masses.sum()) * 9.81)

    def _start_pitch(self):
        return float((self.data.qpos[self.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)      # calm_frac is 0: a harsh (collapsed) start
        self._start_kind = "lying"
        if self._standready and self.np_random.random() < self._crouch_frac:
            self._calm_frac = 1.0                                   # crouched calm start, resampled until clearly crouched
            for _ in range(_MAX_RESAMPLE):
                obs, info = super().reset(options=options)
                if self._knee_min_deg() < CROUCH_KNEE_DEG or self.data.xpos[self.a_torso_id][2] < CROUCH_TORSO_Z:
                    break
            self._calm_frac = 0.0
            self._start_kind = "crouch"
        else:
            want_neg = self._prone_frac > 0.0 and self.np_random.random() < self._prone_frac
            want_pos = (not want_neg) and self._supine_frac > 0.0 and self.np_random.random() < self._supine_frac
            want_flat = self._flat_frac > 0.0 and self.np_random.random() < self._flat_frac
            for _ in range(_MAX_RESAMPLE):
                if self.data.xpos[self.a_torso_id][2] < FLOOR_START_Z and (not want_neg or self._start_pitch() <= 0.0) \
                        and (not want_pos or self._start_pitch() > 0.0) \
                        and (not want_flat or float(np.mean(self.data.qpos[self._hip_adr])) < self._flat_hip):
                    break
                obs, info = super().reset(options=options)
        self._start_side = "neg" if self._start_pitch() <= 0.0 else "pos"
        head = float(self.data.xpos[self.a_head_id][2])
        self._floor_head_best = min(head, HANDOFF_HEAD_Z) / HANDOFF_HEAD_Z
        self._floor_head_max = head
        self._hold = 0
        self._prev_head_z = head
        self._arr_prev_com_x = None
        return obs, info

    def catch_rollout(self, stand_model, n=150):
        """Runs `stand_model` (deterministic) from the CURRENT state for n steps and returns True if it catches it (benchmark criterion); the env state is consumed."""
        obs = self._obs()
        run = 0
        for _ in range(n):
            a, _ = stand_model.predict(obs, deterministic=True)
            obs, *_ = super().step(a)
            z = float(self.data.xpos[self.a_torso_id][2])
            if z < 0.45:
                return False
            pitch = (self.data.qpos[self.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi
            ok = z >= 0.7 and float(self.data.xpos[self.a_head_id][2]) >= 0.9 and abs(float(pitch)) < np.deg2rad(40)
            run = run + 1 if ok else 0
            if self._catch_early and run >= self._catch_early:      # STANDAI_CATCH_EARLY=<steps>: already held that long -> caught, stop the rollout (speed; the trainer waits for the slowest env)
                return True
        return run >= 20

    def step(self, action):
        obs, _parent_reward, terminated, truncated, info = super().step(action)
        head = float(self.data.xpos[self.a_head_id][2])
        foot_down, _ = self._getup_contacts()
        self._floor_head_max = max(self._floor_head_max, head)
        phi = min(head, HANDOFF_HEAD_Z) / HANDOFF_HEAD_Z

        record = 0.0
        if foot_down and phi > self._floor_head_best:
            record = RECORD_K * (phi - self._floor_head_best)
            self._floor_head_best = phi
        torso = float(self.data.xpos[self.a_torso_id][2])
        dense = (DENSE_SCALE * phi + TORSO_DENSE_SCALE * phi * min(torso / 0.7, 1.0)) * foot_down
        pitch = (self.data.qpos[self.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi
        settle = 0.0
        sr_dense = 0.0
        speed_pen = 0.0
        air_pen = 0.0
        sv = 0.0
        v_norm = 0.0
        feet_gap, gap_lim = self._feet_gap_now()
        if self._standready:
            vz = float(self.data.qvel[self.a_z_dof])
            pr = float(np.degrees(self.data.qvel[self._ry_dof]))
            knee = self._knee_min_deg()
            t_z, t_knee, t_pitch, t_vz, t_pr = self._sr_thresholds()
            in_pose = bool(foot_down and torso >= t_z and knee >= -t_knee
                           and abs(pitch) < np.deg2rad(t_pitch) and abs(vz) < t_vz and abs(pr) < t_pr
                           and (self._com_max is None or self._com_ahead() <= self._com_limit())
                           and (gap_lim is None or feet_gap <= gap_lim))
            ext = min(1.0, max(0.0, 1.0 + knee / 60.0))              # 1 at straight knees, 0 at -60 deg or more bent
            sr_dense = SR_DENSE_SCALE * min(1.0, max(0.0, (torso - 0.6) / 0.5)) * ext * foot_down
            if gap_lim is not None:
                sr_dense *= float(np.exp(-(feet_gap / 0.5) ** 2))
            if self._sv_model is not None and foot_down and torso >= 0.7:
                if self._sv_sym:
                    v_stand = self._sv_symp.value(obs)
                else:
                    with self._sv_torch.no_grad():
                        v_stand = float(self._sv_model.policy.predict_values(self._sv_torch.as_tensor(obs[None], dtype=self._sv_torch.float32))[0, 0])
                v_norm = min(1.0, max(0.0, (v_stand - self._sv_lo) / (self._sv_hi - self._sv_lo)))
                # ramp with the head height: no weight below 0.9 m (V of a low, lying-ish state is a poor guess), full weight from 1.3 m (the hand-off height)
                sv = self._sv_coef * min(1.0, max(0.0, (head - 0.9) / 0.4)) * v_norm
            if self._speed_pen > 0.0:
                h_w = min(1.0, max(0.0, (torso - SP_H_LO) / (SP_H_HI - SP_H_LO)))
                speed_pen = self._speed_pen * h_w * (self._speed_vz_w * abs(vz) / SR_MAX_VZ + abs(pr) / SR_MAX_PITCH_RATE)
            if self._air_pen > 0.0 and torso > SP_H_LO and not foot_down:
                air_pen = self._air_pen
            if foot_down and torso >= 0.9:
                settle = SR_SETTLE_SCALE * float(np.exp(-(vz / SR_MAX_VZ) ** 2 - (pr / SR_MAX_PITCH_RATE) ** 2))
        elif self._settled:
            vz = float(self.data.qvel[self.a_z_dof])
            pr = float(np.degrees(self.data.qvel[self._ry_dof]))
            in_pose = bool(foot_down and head >= HANDOFF_HEAD_Z and torso >= SETTLED_TORSO_Z
                           and abs(pitch) < np.deg2rad(SETTLED_MAX_PITCH_DEG)
                           and abs(vz) < SETTLED_MAX_VZ and abs(pr) < SETTLED_MAX_PITCH_RATE)
            if foot_down and head >= 0.8 * HANDOFF_HEAD_Z and torso >= HOLD_TORSO_Z:
                settle = SETTLE_SCALE * float(np.exp(-(vz / SETTLED_MAX_VZ) ** 2 - (pr / SETTLED_MAX_PITCH_RATE) ** 2))
        else:
            # v4-type "reach" target (non-standready); with STANDAI_FEET_GAP the same level curriculum tightens a feet-height-gap limit here too
            in_pose = bool(foot_down and head >= HANDOFF_HEAD_Z and torso >= HOLD_TORSO_Z
                           and abs(pitch) < np.deg2rad(HANDOFF_MAX_PITCH_DEG)
                           and (gap_lim is None or feet_gap <= gap_lim))
        self._hold = self._hold + 1 if in_pose else 0
        success = self._hold >= (SETTLED_HOLD_STEPS if self._settled and not self._standready else HOLD_STEPS)
        bonus = (SR_BONUS if self._standready else SUCCESS_BONUS) if success else 0.0
        if success and self._sv_model is not None:       # the arrival state is what the stand policy will have to hold: reward the ones it values
            sv += self._sv_bonus * v_norm
        com_x = float(np.sum(self._a_masses * self.data.xipos[self._a_bodies, 0]) / self._a_masses.sum())
        com_vx = 0.0 if self._arr_prev_com_x is None else (com_x - self._arr_prev_com_x) / 0.02
        self._arr_prev_com_x = com_x
        arrival = 0.0
        if success and self._arr_k > 0.0 and self._standready:
            arrival = self._arr_k * (float(np.clip(com_vx, -0.2, self._arr_v)) / self._arr_v - float(np.clip((np.degrees(pitch) - 3.0) / 20.0, 0.0, 1.0)))
        arm_push = 0.0
        if self._arm_coef > 0.0:
            head_vz = (head - (self._prev_head_z if self._prev_head_z is not None else head)) / 0.02
            if torso < 0.7 and head_vz > 0.05:
                arm_push = self._arm_coef * min(self._arm_force_ratio(), 1.5)
        self._prev_head_z = head
        reward = record + dense + sr_dense + settle + bonus - speed_pen - air_pen + sv + arm_push + arrival

        info["reward_breakdown"] = {"head_record": record, "ground_height": dense, "arm_push": arm_push, "arrival": arrival}
        if self._settled or self._standready:
            info["reward_breakdown"]["settle"] = settle
        if self._standready:
            info["reward_breakdown"]["standready_dense"] = sr_dense
            info["reward_breakdown"]["speed_pen"] = -speed_pen
            info["reward_breakdown"]["air_pen"] = -air_pen
            info["reward_breakdown"]["stand_value"] = sv
        if success:
            terminated = True
            info["handoff_state"] = (self.data.qpos.copy(), self.data.qvel.copy(), head)
        if self.step_count >= self._max_steps:
            truncated = True
        if terminated or truncated:
            info["floor_success"] = success
            info["start_kind"] = self._start_kind
            info["start_side"] = self._start_side
            info["floor_level"] = self._level
            info["floor_head_max"] = self._floor_head_max
        if success and self._catch_model is not None and self._standready:
            caught = self.catch_rollout(self._catch_model, self._catch_n)
            reward += self._catch_k * float(caught)
            info["reward_breakdown"]["catch"] = self._catch_k * float(caught)
            info["catch"] = bool(caught)
        return obs, reward, terminated, truncated, info
