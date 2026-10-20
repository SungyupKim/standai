"""Every N minutes, one Kakao message about whatever the overnight pipelines are doing right now: finds the log files that were written in the last 20 minutes
(floor / stand training logs, distillation, evaluation outputs), and reports the stage, steps, training success / level where there are any, and the last lines of an evaluation.
It does not touch the pipelines. Stops when no pipeline script is running any more (after one last message).
Usage: python heartbeat.py [--interval-minutes 30] [--once] [--dry]"""
import argparse, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CK = os.path.join(HERE, "..", "checkpoints") + "/"
# log -> (label, total steps shown, includes the initial counter of the warm-started model)
TRAIN = [("floor_train_bigp2.log", "엎드린 큰 네트워크 2단계(bigp2)", 35_000_000), ("floor_train_bigp1.log", "엎드린 큰 네트워크 1단계(bigp1)", 10_000_000),
         ("floor_train_bigd.log", "누운 큰 네트워크, 처음부터 최종 목표(bigd)", 20_000_000), ("floor_train_bigdA.log", "누운 bigd 이어서 학습(갈래 A)", None),
         ("locomotion_train_stand_big1.log", "서기 정책 큰 네트워크 미세조정(갈래 B)", 50_000_000)]
OTHER = [("distill_stand.log", "서기 정책 증류(갈래 B)"), ("analysis/ctl_prone_bigp2.out", "엎드린 bigp2 측정"), ("analysis/ctl_supine_bigd.out", "누운 bigd 측정"),
         ("analysis/ctl_supine_bigdA.out", "누운 갈래 A 측정"), ("analysis/ladder_B_final.out", "갈래 B 최종 측정"), ("analysis/floor_ckpt_scan_bigdA.out", "체크포인트 스캔")]


def last_val(path, key):
    v = None
    for line in open(path, errors="ignore"):
        m = re.match(rf"\|\s+{re.escape(key)}\s+\|\s+([-\d.e+]+)", line)
        if m:
            v = float(m.group(1))
    return v


def last_ladder_or_level(path):
    out = None
    for line in open(path, errors="ignore"):
        if "curriculum level" in line or "COM-ahead ladder at" in line:
            out = line.strip()[:140]
    return out


def describe():
    now, parts = time.time(), []
    for f, label, total in TRAIN:
        p = CK + f
        if os.path.exists(p) and now - os.path.getmtime(p) < 20 * 60:
            ts, succ, lvl = last_val(p, "total_timesteps"), None, last_val(p, "level")
            for line in open(p, errors="ignore"):
                m = re.match(r"\|\s+success_rate\s+\|\s+([\d.]+)", line)
                if m:
                    succ = float(m.group(1))
            s = f"[{label}] 스텝 {ts / 1e6:.1f}M" + (f"/{total / 1e6:.0f}M ({100 * ts / total:.0f}%)" if total and ts else "")
            if succ is not None: s += f" | 학습 성공률 {succ:.2f}"
            if lvl is not None: s += f" | 목표 레벨 {lvl:.2f}"
            kl, std = last_val(p, "approx_kl"), last_val(p, "std")
            if kl is not None: s += f" | KL {kl:.3f}"
            if std is not None: s += f", std {std:.2f}"
            if "stand" in f:
                fr = last_val(p, "ep_rew_mean")
                if fr is not None: s += f" | ep_rew {fr:.0f}"
            parts.append(s)
    for f, label in OTHER:
        p = CK + f
        if os.path.exists(p) and now - os.path.getmtime(p) < 20 * 60:
            tail = [l.strip()[:150] for l in open(p, errors="ignore").read().splitlines() if l.strip()][-3:]
            parts.append(f"[{label}] " + " / ".join(tail))
    return parts


def pipelines_alive():
    out = subprocess.run(["pgrep", "-af", "overnight_"], capture_output=True, text=True).stdout
    return any("overnight_" in l and "pgrep" not in l and "bash -c" not in l for l in out.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-minutes", type=float, default=30)
    ap.add_argument("--once", action="store_true"); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    from kakao_notify import send_message
    idle = 0
    while True:
        parts = describe()
        alive = pipelines_alive()
        msg = f"[{time.strftime('%H:%M')} 경과] " + ("\n".join(parts) if parts else "지금 쓰는 로그 없음(단계 전환 중이거나 대기)") + ("" if alive else "\n(파이프라인 스크립트가 모두 끝남)")
        print(msg, flush=True)
        if not a.dry:
            try:
                send_message(msg)
            except Exception as e:
                print("send failed:", e, flush=True)
        if a.once or (not alive and idle >= 1):
            break
        idle = idle + 1 if not alive else 0
        time.sleep(a.interval_minutes * 60)


if __name__ == "__main__":
    main()
