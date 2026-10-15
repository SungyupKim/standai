"""KakaoTalk progress notifier for the floor get-up run (train_floor.py), same idea as
stand_progress_notify.py: the training-log numbers are stochastic (noise + env-level bursts), so every
interval the latest autosave is also measured deterministically on fixed seeds, policy alone."""
import argparse
import glob
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from kakao_notify import send_message

EVAL_N = 50
FIELDS = {
    "total_timesteps": "누적 스텝",
    "success_rate": "성공률(학습 로그, 노이즈+버스트 포함)",
    "head_max_mean": "머리 최대 높이 평균(학습 로그)",
    "head_over_0.5": "머리 0.5m 넘은 비율(학습 로그)",
    "std": "행동 흔들림(std)",
    "approx_kl": "approx_kl",
    "level": "목표 문턱 단계(0 느슨 ~ 1 엄격, 학습 중)",
}
LINE_RE = re.compile(r"\|\s*([\w/.]+)\s*\|\s*([-+\d.eE]+)\s*\|")


def latest_values(log_path):
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 60_000))
        tail = f.read().decode(errors="ignore").splitlines()
    values = {}
    for line in reversed(tail):
        m = LINE_RE.search(line)
        if not m:
            continue
        key = m.group(1).split("/")[-1]
        if key in FIELDS and key not in values:
            values[key] = float(m.group(2))
        if len(values) == len(FIELDS):
            break
    return values


def finished(log_path):
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4000))
        return "saved model to" in f.read().decode(errors="ignore")


def latest_checkpoint(ckpt_glob):
    files = glob.glob(ckpt_glob)
    return max(files, key=os.path.getmtime) if files else None


def policy_alone_eval(ckpt_path, n=EVAL_N):
    """Deterministic actions, no noise or bursts, seeds 0..n-1: (success rate, share with head > 0.5 m,
    share with head > 0.8 m, mean peak head height)."""
    from stable_baselines3 import PPO
    from floor_env import FloorEnv
    model = PPO.load(ckpt_path, device="cpu")
    env = FloorEnv()
    succ, peaks = 0, []
    for seed in range(n):
        obs, _ = env.reset(seed=seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        succ += bool(info["floor_success"])
        peaks.append(info["floor_head_max"])
    peaks = sorted(peaks)
    return (succ / n, sum(p > 0.5 for p in peaks) / n, sum(p > 0.8 for p in peaks) / n, sum(peaks) / n)


def standready_eval(ckpt_path, stand_ckpt, n=60, crouch_frac=0.4, max_steps=None, prone_frac=0.0, down_timeout=None, level=1.0, com_max=None):
    """v6: deterministic floor policy on the mixed start distribution (lying + crouched), fixed seeds 0..n-1. When it reaches
    the stand-ready region the stand policy takes over for 3 s (gate: torso never below 0.7 m). Returns per start kind
    {kind: [starts, reached, held_by_stand]}."""
    import numpy as np
    from stable_baselines3 import PPO
    from env import FALL_HEIGHT
    from floor_env import FloorEnv, SR_MAX_STEPS
    model = PPO.load(ckpt_path, device="cpu")
    stand = PPO.load(stand_ckpt, device="cpu") if stand_ckpt else None
    env = FloorEnv(standready=True, crouch_frac=crouch_frac, max_steps=max_steps, prone_frac=prone_frac,
                   down_timeout=down_timeout, level=level, com_max=com_max)
    out = {"lying": [0, 0, 0], "crouch": [0, 0, 0], "_pos": [0, 0, 0], "_neg": [0, 0, 0]}   # "_" rows: lying starts by pitch side
    for seed in range(n):
        obs, _ = env.reset(seed=seed)
        kind = env._start_kind
        side = "_" + env._start_side if kind == "lying" else None
        out[kind][0] += 1
        if side:
            out[side][0] += 1
        for _ in range(max_steps or SR_MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            if info.get("floor_success"):
                out[kind][1] += 1
                if side:
                    out[side][1] += 1
                if stand is not None:
                    fell = False
                    for _ in range(150):
                        action, _ = stand.predict(obs, deterministic=True)
                        obs, *_ = env.step(action)
                        if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT:
                            fell = True
                            break
                    out[kind][2] += (not fell)
                    if side:
                        out[side][2] += (not fell)
                break
            if term or trunc:
                break
    return out


def format_standready(v, total, ev, ckpt, baseline_steps=0, ev_level=None):
    steps = v.get("total_timesteps", 0)
    pct = max(0.0, min(100.0, 100 * (steps - baseline_steps) / total)) if total else 0
    lines = ["[일어나기 v6 학습 진행 (서 있기 영역까지 올려서 인계)]",
             f"스텝: {steps / 1e6:.2f}M (이번 구간 {pct:.0f}%)"]
    for k in ("success_rate", "std", "approx_kl", "level"):
        if k in v:
            lines.append(f"{FIELDS[k]}: {v[k]:.3f}")
    if ev_level is not None:
        n_l = sum(x[0] for k, x in ev_level.items() if not k.startswith("_"))
        r_l = sum(x[1] for k, x in ev_level.items() if not k.startswith("_"))
        h_l = sum(x[2] for k, x in ev_level.items() if not k.startswith("_"))
        lines.append(f"현재 문턱 단계 {v.get('level', 1.0):.2f}에서 도달: {r_l}/{n_l}, 이어받은 서 있기가 3초 버팀 {h_l}/{r_l} (엄격 단계 1.0 기준은 아래)")
    if ev is not None:
        main = {k: x for k, x in ev.items() if not k.startswith("_")}
        tot = sum(x[0] for x in main.values()); reached = sum(x[1] for x in main.values()); held = sum(x[2] for x in main.values())
        lines.append(f"정책 단독(결정적, 고정 시드 {tot}개) 서 있기 영역 도달: {reached}/{tot} ({reached / max(tot, 1):.0%})")
        for kind, (n, r, h) in main.items():
            if n == 0:
                continue
            lines.append(f"  {kind}: 도달 {r}/{n}, 이어받은 서 있기가 3초 버팀 {h}/{r}")
        for side, label in (("_pos", "시작 기울기 +쪽(누움, 약 80%)"), ("_neg", "시작 기울기 -쪽(엎드림 추정, 약 20%)")):
            if side in ev and ev[side][0]:
                n, r, h = ev[side]
                lines.append(f"  {label}: 도달 {r}/{n} ({r / n:.0%}), 버팀 {h}/{r}")
        lines.append(f"끝에서 끝까지(도달 후 3초 버팀): {held}/{tot} ({held / max(tot, 1):.0%})")
        lines.append(f"  기준 체크포인트: {os.path.basename(ckpt)}")
    return "\n".join(lines)


def format_message(v, total, ev, ckpt, baseline_steps=0):
    steps = v.get("total_timesteps", 0)
    done = max(0.0, steps - baseline_steps)
    lines = ["[바닥 일어나기 학습 진행]",
             f"스텝: {steps / 1e6:.2f}M / {(baseline_steps + total) / 1e6:.0f}M ({done / total:.0%})"]
    for k in ("success_rate", "head_max_mean", "head_over_0.5", "std", "approx_kl"):
        if k in v:
            lines.append(f"{FIELDS[k]}: {v[k]:.3f}")
    if ev is not None:
        s, h5, h8, hm = ev
        lines.append(f"정책 단독(결정적, 고정 시드 {EVAL_N}개): 성공 {s:.0%} / 머리>0.5m {h5:.0%} / "
                     f"머리>0.8m {h8:.0%} / 최대높이 평균 {hm:.2f}m")
        lines.append(f"  기준 체크포인트: {os.path.basename(ckpt)}")
    else:
        lines.append("정책 단독 측정: 체크포인트 없음")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--ckpt-glob", required=True)
    parser.add_argument("--total-timesteps", type=float, required=True)
    parser.add_argument("--interval-minutes", type=float, default=30)
    parser.add_argument("--standready", action="store_true", help="evaluate the v6 stand-ready target (mixed lying/crouch starts)")
    parser.add_argument("--eval-n", type=int, default=60, help="fixed seeds per --standready evaluation (60 was too noisy: +-6 points)")
    parser.add_argument("--crouch-frac", type=float, default=0.4, help="share of crouched starts in --standready evaluation (0 = lying only)")
    parser.add_argument("--max-steps", type=int, default=None, help="episode length for the --standready evaluation (match the training run)")
    parser.add_argument("--prone-frac", type=float, default=0.0, help="share of lying evaluation starts forced to the start-pitch <= 0 side (1.0 = prone-only policy)")
    parser.add_argument("--com-max", type=float, default=None, help="evaluate with the COM-over-front-foot condition (match the training run)")
    parser.add_argument("--no-down-timeout", action="store_true", help="evaluate without env.py's 5 s 'stayed down' KO (match the training run)")
    parser.add_argument("--stand-ckpt", default=None, help="stand policy that takes over in --standready evaluation")
    parser.add_argument("--baseline-steps", type=float, default=0.0,
                        help="cumulative steps when this launch started (warm start), so the progress % is for this leg")
    args = parser.parse_args()
    log_path = pathlib.Path(args.log)
    last_steps = None
    while True:
        try:
            v = latest_values(log_path)
            fin = finished(log_path)
            if v and (v.get("total_timesteps") != last_steps or fin):
                last_steps = v.get("total_timesteps")
                ckpt = latest_checkpoint(args.ckpt_glob)
                if args.standready:
                    ev = standready_eval(ckpt, args.stand_ckpt, n=args.eval_n, crouch_frac=args.crouch_frac,
                                          max_steps=args.max_steps, prone_frac=args.prone_frac,
                                          down_timeout=not args.no_down_timeout, com_max=args.com_max) if ckpt else None
                    lvl = v.get("level", 1.0)
                    ev_lv = standready_eval(ckpt, args.stand_ckpt, n=args.eval_n, crouch_frac=args.crouch_frac,
                                            max_steps=args.max_steps, prone_frac=args.prone_frac,
                                            down_timeout=not args.no_down_timeout, level=lvl, com_max=args.com_max) if ckpt and lvl < 1.0 else None
                    msg = format_standready(v, args.total_timesteps, ev, ckpt, args.baseline_steps, ev_level=ev_lv)
                else:
                    ev = policy_alone_eval(ckpt) if ckpt else None
                    msg = format_message(v, args.total_timesteps, ev, ckpt, args.baseline_steps)
                send_message(msg + ("\n\n학습 완료 (모델 저장됨)" if fin else ""))
                print(f"[floor-notify] sent at {last_steps:.0f} steps (alone={ev})", flush=True)
            if fin:
                print("[floor-notify] finished", flush=True)
                return
        except Exception as e:
            print(f"[floor-notify] failed: {e}", flush=True)
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
