"""Every N minutes: measure the newest autosave of a floor run (deterministic, fixed seeds, hand-over region reached = success rate) and send a Kakao message with the trend.
The training log of a distilled student has no tables, so this is the only progress number for such runs. Same measurement as analysis/floor_ckpt_scan.py (seeds 25000+, level 0.7).
Usage: nice -n 10 python floor_trend_notify.py --glob '../checkpoints/autosave/ppo_floor_fdist2_*_steps.zip' --label fdist2 --total-steps 140000000 --ref 70.5 --ref-label 'fdist 24~39M 평균'"""
import argparse, glob, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "analysis"))
from kakao_notify import send_message
from floor_ckpt_scan import reach

key = lambda f: int(re.search(r"_(\d+)_steps", f).group(1))


def measure(ck, side, n, seed0, level, workers):
    parts = [(ck, side, n // workers, seed0 + i * (n // workers), level, 450) for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        ok = [x for _, o in ex.map(reach, parts) for x in o]
    return sum(ok), len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--total-steps", type=float, required=True, help="step counter at which the run ends (includes the warm-start counter)")
    ap.add_argument("--interval-minutes", type=float, default=30); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=25000); ap.add_argument("--side", default="supine"); ap.add_argument("--level", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--ref", type=float, default=None); ap.add_argument("--ref-label", default="기준")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    hist_path = os.path.join(HERE, "..", "checkpoints", f"trend_{a.label}.json")
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    while True:
        files = glob.glob(a.glob)
        if files:
            ck = max(files, key=key); st = key(ck)
            if not hist or hist[-1][0] != st:
                ok, n = measure(ck, a.side, a.n, a.seed0, a.level, a.workers)
                hist.append([st, ok, n]); json.dump(hist, open(hist_path, "w"))
            trend = " → ".join(f"{h[1] / h[2]:.0%}" for h in hist[-6:])
            last = hist[-1]; pct = last[1] / last[2]
            lines = [f"[{a.label}] 누운 바닥 정책 이어서 학습", f"스텝 {last[0] / 1e6:.1f}M / {a.total_steps / 1e6:.0f}M ({last[0] / a.total_steps:.0%})",
                     f"인계 영역 도달 {last[1]}/{last[2]} = {pct:.1%} (고정 시드 {a.seed0}~, 레벨 {a.level}, 결정적 행동)", f"최근 추세: {trend}"]
            if a.ref is not None:
                lines.append(f"{a.ref_label} {a.ref:.1f}% (측정 오차 ±2.7%p, 체크포인트 간 흔들림 ±4%p)")
            send_message("\n".join(lines))
        if a.once or (files and key(max(files, key=key)) >= a.total_steps):
            break
        time.sleep(a.interval_minutes * 60)


if __name__ == "__main__":
    main()
