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
                ev = policy_alone_eval(ckpt) if ckpt else None
                msg = format_message(v, args.total_timesteps, ev, ckpt)
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
