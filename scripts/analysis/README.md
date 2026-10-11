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

`getup_cem.py` / `getup_mpc.py` replicate `env.step`'s stand-only physics in a light rollout
(verified identical to `env.step`, max |dz| = 0.0000). The MPC is a *diagnostic*, not a teacher: it
queries the exact simulator, so its success says what the body can do, not what a policy has learned.
