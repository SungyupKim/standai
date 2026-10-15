# scripts/analysis

Diagnostics written during the 2026-10-11 stand-only investigation (see
`docs/experiments_2026-10-11.md`). Run all of them from the repo root with `.venv/bin/python`.
Outputs go to `checkpoints/analysis/` (gitignored).

| script | what it answers |
|---|---|
| `gate_one.py <ckpt>` | calm / full / harsh-only deterministic 3s gate (100 seeds) |
| `calm_knee_table.py [ckpt]` | calm success vs initial knee bend and torso height (300 seeds) |
| `harsh_diag.py [ckpt]` | does the policy get up from harsh starts at all? height gain, by start height |
| `reward_by_height.py [ckpt] [--ground-shaping\|--getup\|--knee-gate]` | per-step reward terms by torso height (found the reward valley) |
| `tumble_diag.py <ckpt>` | how much the torso rotates on harsh starts, and whether episodes end *genuinely* upright |
| `lying_strategy.py <ckpt>` | while lying: floor contacts, knee height, hip swing, knee_avoid reward |
| `chain_probe.py [ckpt]` | can the rise be CONTINUED from mid-height poses (policy vs random bursts), and does it end upright |
| `random_explore.py [ckpt]` | policy-FREE random motion (AR noise / held bang-bang) vs the policy on floor starts: how high does the head get |
| `explore_probe.py <ckpt>` | does more / temporally-correlated action noise raise how often the head gets up from the floor |
| `gear_policy_eval.py <scale> [ckpt]` | the same policy on a stronger/weaker body |
| `getup_cem.py` | open-loop CEM trajectory search (final objective, v4). Mostly a negative result |
| `getup_mpc.py` | closed-loop predictive-sampling MPC oracle; `GEAR_SCALE=2 python scripts/analysis/getup_mpc.py` |

| `mpc_hold.py` | can ANY controller keep the body up 3 s from a given state (hand-off states, crouched calm starts)? |
| `stand_velocity_map.py` | how much torso vz / pitch rate the stand policy tolerates on upright starts (used to define the stand-ready region) |
| `feet_only_gate.py` | contact-based "standing" definition (decomposed; its 0/300 "every step" result is an artefact, do not cite) |

Floor-policy tools (2026-10-13..15, see experiments doc 3.17). Several hard-code `/home/ubuntu/workspace/standai` and the
stand policy `ppo_stand_v37_ladder_..._96000000_steps.zip`; edit the constants at the top when moving the repo.

| script | what it answers |
|---|---|
| `speed_profile.py [ckpt] [n]` | floor policy: |pitch rate| / |vz| by torso height (the braking habit; baseline 53% of steps above 1.0 m below 100 deg/s) |
| `floor_stuck_diag.py` | which stand-ready condition fails at the highest-torso step |
| `floor_miss_classify.py <ckpt> [n] [max_steps] [prone_frac]` | why starts do NOT reach the region: never rose / rose partway / reached 1.08 m but not all six conditions |
| `peak_posture.py <ckpt> [n]` | torso pitch, waist, knee at the highest-torso step (waist sat at its +45 deg limit) |
| `time_limit_probe.py <ckpt> [n] [max_steps]` | reach / held-by-stand vs episode length (300 / 450 / 600 / 750 steps) |
| `eval_ckpt.py <ckpt> <n>` | prone-only: reached and held-by-stand on n fixed seeds (strict region) |
| `eval_com.py <ckpt> <com_max> <n>` | the same with the centre-of-mass-over-front-foot condition |
| `handoff_features.py <ckpt> [n]` | hand-off states the stand policy holds vs not: feature means |
| `handoff_auc.py <ckpt> [n]` | the same ranked by AUC (COM x - front foot x was 0.26, everything else <= 0.4 or >= 0.6 weak) |
| `hold_compare.py <ckpt> <n> <prone_frac>` | after reaching the region: stand policy v37 vs the floor policy ITSELF holding 3 s |
| `floor_only_hold.py <ckpt> <n> <prone_frac>` | floor policy alone: never-below-0.7 m vs torso >= 0.7 m at 3 s / 4 s (different criteria, very different numbers) |
| `system_loop.py <A\|B\|C> <n> [steps]` | whole-system structures over 20 s: A floor<->stand loop, B floor only, C floor->stand without a way back |
| `mpc_prone.py <seeds…>` | MPC on prone-side starts (5/6 stand), wrapper that does not overwrite the saved .npz |
| `best_ckpt_guard.py` | keeps a copy of the best-evaluated checkpoint while a run trains (never touches training) |

`getup_cem.py` / `getup_mpc.py` replicate `env.step`'s stand-only physics in a light rollout
(verified identical to `env.step`, max |dz| = 0.0000). The MPC is a *diagnostic*, not a teacher: it
queries the exact simulator, so its success says what the body can do, not what a policy has learned.
