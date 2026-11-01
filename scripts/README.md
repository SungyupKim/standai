# scripts/

Run everything from the repo root with `.venv/bin/python scripts/<name>.py` (the shell pipelines `cd` to the right place themselves). Outputs, logs and checkpoints go to `checkpoints/` (gitignored). Why each piece exists and what it showed: [`docs/TECHNICAL_SUMMARY.md`](../docs/TECHNICAL_SUMMARY.md), [`docs/experiments_2026-10-11.md`](../docs/experiments_2026-10-11.md).

## Start here (the current best system: one network)

| script | what |
|---|---|
| `demo_hf.py` | quick start: downloads the single network from Hugging Face, measures it, `--video` renders a clip |
| `getup_env.py` | `GetUpEnv`: the whole get-up task as one 20 s environment (random supine/prone start, no termination at the hand-over) |
| `distill_controller.py` | DAgger: distil the whole rule-based controller (`controller.py`) into one network |
| `train_getup.py` | end-to-end PPO fine-tune of that network on the 20 s standing reward (+ value warm-up) |
| `analysis/getup_strict_eval.py`, `getup_eval.py` | paired evaluation against the rule-based controller on the same seeds (strict criteria / benchmark criterion) |
| `render_demo.py` | headless demo video/GIF (`--student <zip>` for the single network, `--scale 2 --crf 16` for 1280x720) |

## The pieces of the rule-based controller (teacher and baseline)

| script | what |
|---|---|
| `controller.py` | `GetUpController`: floor policy (by start pitch sign) -> hand-over policy (prone only) -> stand policy, fall back to the floor policy; opt-in diagnostics `STANDAI_VX_GATE`, `STANDAI_CLF_GATE` |
| `locomotion_env.py`, `train_locomotion.py` | stand environment / trainer (curricula, hand-off pools, COM-ahead ladder) |
| `floor_env.py`, `train_floor.py` | floor get-up environment (lying starts, hand-over region level, arrival and catch rewards) / trainer |
| `handover_env.py`, `train_handover.py` | hand-over policy environment / trainer (warm start from the stand policy) |
| `distill_stand.py`, `distill_floor.py` | 64x64 -> 256x256 distillation of a stand / floor policy (behaviour cloning + value regression + DAgger) |
| `make_handoff_pool.py`, `make_strict_pool.py` | collect hand-over states (`--supine-only`, `--level`) for training and analysis |
| `sym.py` | left/right symmetrised stand policy (`STANDAI_SYM_STAND=1`) |

## Shared body / environment base (copied from fightai)

`build_model.py`, `env.py`, `view.py`, `kakao_notify.py` (a body change needs the same edit in both repos), `viewer_utils.py`.

## Viewers

`watch_controller.py` (whole rule-based system), `watch_floor.py`, `watch_locomotion.py`, `watch_failures.py`. They need an OpenGL display (`DISPLAY=:1` on this machine).

## Progress notifiers (KakaoTalk)

`heartbeat.py` (30-minute pipeline message), `stand_progress_notify.py`, `floor_progress_notify.py`, `floor_trend_notify.py` (reach rate / arrival speed / catch rate for runs whose log has no table), `handover_progress_notify.py`, `getup_trend_notify.py` (single-network runs).

## pipelines/ (unattended shell pipelines: train -> scan -> compare -> Kakao message)

The first block is the recipe of the current system; the rest are records of experiments (mostly negative) that are kept so they can be reproduced. They `cd` to the repo and call the scripts above; they do not export training variables (`env VAR=... cmd`), and the measurements run with `env -u`.

| pipeline | role |
|---|---|
| `overnight_ladder.sh` | stand policy: distilled 256x256 + PPO with the COM-ahead ladder (`best_stand_big1`) |
| `overnight_floor_distill.sh` | supine floor policy: distil v12 into 256x256 + PPO (fdist) |
| `overnight_floor_more.sh` | supine floor policy: +100M steps (fdist2) |
| `overnight_floor_arrival.sh` | supine floor policy: continuous arrival-quality reward (arr1, the one that worked) |
| `overnight_handover_big.sh` | prone hand-over policy re-trained from the big stand policy (h4) |
| `overnight_size_ablation.sh` | network size 256/128/64 of the single network (distil + 12M end-to-end steps) |
| `overnight_big.sh`, `overnight_big_direct.sh`, `overnight_big_prone.sh` | big floor policies from scratch (failed) |
| `overnight_s2.sh`, `overnight_handover_supine_big.sh` | supine hand-over policies s2, s3 (no gain) |
| `overnight_stand_big2.sh` | stand policy re-trained on the new supine arrivals (no gain in the controller) |
| `overnight_arr2.sh`, `overnight_arr2_resume.sh` | stronger arrival reward (no gain); the resume script handles 0-byte autosaves after a reboot |
| `overnight_floor_catch.sh`, `overnight_floor_catch2.sh`, `catch1_15m.sh` | stand-policy catch outcome as floor reward (not significant) |
| `clf_gate_sweep.sh`, `clf_gate_confirm.sh` | learned hand-over gate sweep and confirmation (+2-3 points, off by default) |
| `after_ap1.sh`, `after_flat1.sh`, `after_sv1.sh` | measurements after older supine floor fine-tunes (arm push, flat starts, critic aux) |

## analysis/

Measurements and diagnostics behind the docs; see [`analysis/README.md`](analysis/README.md).
