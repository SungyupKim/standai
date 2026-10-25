"""Every N minutes: measure the newest autosave of an end-to-end get-up run (standing at 10 s / 20 s per start side, fixed seeds, deterministic) and send a Kakao message with the trend, next to the rule-based
controller measured once on the same seeds. Usage: nice -n 10 python getup_trend_notify.py --glob '../checkpoints/autosave/ppo_getup_e2e1_*_steps.zip' --label e2e1 --total-steps 40000000"""
import argparse, glob, json, os, re, sys, time
from multiprocessing import get_context
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from distill_controller import eval_pool
from kakao_notify import send_message

key = lambda f: int(re.search(r"_(\d+)_steps", f).group(1))


def ratios(r):
    s, p = r["supine"], r["prone"]
    return s[1] / s[2], p[1] / p[2], (s[1] / s[2] + p[1] / p[2]) / 2, (s[0] / s[2] + p[0] / p[2]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True); ap.add_argument("--label", required=True); ap.add_argument("--total-steps", type=float, required=True)
    ap.add_argument("--interval-minutes", type=float, default=30); ap.add_argument("--n", type=int, default=96); ap.add_argument("--seed0", type=int, default=93000); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    hp = os.path.join(HERE, "..", "checkpoints", f"trend_{a.label}.json")
    hist = json.load(open(hp)) if os.path.exists(hp) else []
    with get_context("fork").Pool(a.workers) as pool:
        base = ratios(eval_pool(pool, None, a.n, a.seed0, a.workers))
        while True:
            files = [f for f in glob.glob(a.glob) if os.path.getsize(f) > 0]
            if files:
                ck = sorted(files, key=key)[-2] if len(files) > 1 else files[0]
                st = key(ck)
                if not hist or hist[-1][0] != st:
                    hist.append([st, *ratios(eval_pool(pool, ck, a.n, a.seed0, a.workers))]); json.dump(hist, open(hp, "w"))
                h = hist[-1]; tr = " → ".join(f"{x[3]:.0%}" for x in hist[-6:])
                send_message(f"[{a.label}] 단일 네트워크 끝에서 끝까지 학습\n스텝 {h[0] / 1e6:.1f}M / {a.total_steps / 1e6:.0f}M\n20초 서기: 누운 {h[1]:.0%}, 엎드린 {h[2]:.0%}, 50:50 {h[3]:.1%} (10초 {h[4]:.0%})\n규칙 컨트롤러(같은 시드 {a.seed0}~, {a.n}/side): 누운 {base[0]:.0%}, 엎드린 {base[1]:.0%}, 50:50 {base[2]:.1%}\n최근 50:50 추세: {tr}")
            if hist and hist[-1][0] >= a.total_steps:
                break
            time.sleep(a.interval_minutes * 60)


if __name__ == "__main__":
    main()
