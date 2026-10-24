"""Every N minutes: measure the newest autosave of a floor run (deterministic, fixed seeds, hand-over region reached = success rate) and send a Kakao message with the trend.
The training log of a distilled student has no tables, so this is the only progress number for such runs. Same measurement as analysis/floor_ckpt_scan.py (seeds 25000+, level 0.7).
Usage: nice -n 10 python floor_trend_notify.py --glob '../checkpoints/autosave/ppo_floor_fdist2_*_steps.zip' --label fdist2 --total-steps 140000000 --ref 70.5 --ref-label 'fdist 24~39M 평균'"""
import argparse, glob, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "analysis"))
from kakao_notify import send_message
from floor_ckpt_scan import reach


def reach_stats(args):
    """Like floor_ckpt_scan.reach, plus the arrival state: COM forward speed (m/s), pitch (deg) and time (s) at the success step."""
    ck, side, n, seed0, level, max_steps = args
    import numpy as np
    from stable_baselines3 import PPO
    from floor_env import FloorEnv
    import torch; torch.set_num_threads(1)
    m = PPO.load(ck, device="cpu")
    stand = PPO.load(STAND_CKPT, device="cpu")
    env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=max_steps, down_timeout=False, level=level, speed_pen=0.08, speed_vz_w=0.25,
                   prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    com = lambda: float(np.sum(env._a_masses * env.data.xipos[env._a_bodies, 0]) / env._a_masses.sum())
    out = []
    for s in range(seed0, seed0 + n):
        obs, _ = env.reset(seed=s); prev = com(); r = None
        for t in range(max_steps):
            a, _ = m.predict(obs, deterministic=True); obs, _, te, tr, info = env.step(a)
            c = com(); vx = (c - prev) / 0.02; prev = c
            if info.get("floor_success"):
                r = (vx, float(np.degrees((env.data.qpos[env.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi)), t * 0.02)
                r = r + (float(env.catch_rollout(stand, 150)),); break
        out.append(r)
    return out

key = lambda f: int(re.search(r"_(\d+)_steps", f).group(1))


ARR_K, ARR_V = 15.0, 0.45
STAND_CKPT = os.path.join(HERE, "..", "checkpoints", "best_stand_big1.zip")


def measure(ck, side, n, seed0, level, workers):
    import numpy as np
    parts = [(ck, side, n // workers, seed0 + i * (n // workers), level, 450) for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        res = [x for o in ex.map(reach_stats, parts) for x in o]
    got = [x for x in res if x is not None]
    vx, pt, tt = (np.array([g[i] for g in got]) for i in range(3)) if got else (np.zeros(1),) * 3
    score = ARR_K * (np.clip(vx, -0.2, ARR_V) / ARR_V - np.clip((pt - 3) / 20, 0, 1))
    return len(got), len(res), [round(float(vx.mean()), 3), round(float(pt.mean()), 2), round(float((vx >= 0.2).mean()), 3), round(float(np.median(tt)), 2), round(float(score.mean()), 2), round(float(np.mean([g[3] for g in got])), 3) if got else 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--total-steps", type=float, required=True, help="step counter at which the run ends (includes the warm-start counter)")
    ap.add_argument("--interval-minutes", type=float, default=30); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=25000); ap.add_argument("--side", default="supine"); ap.add_argument("--level", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--ref", type=float, default=None); ap.add_argument("--ref-label", default="기준")
    ap.add_argument("--once", action="store_true"); ap.add_argument("--baseline-ckpt", default=None); ap.add_argument("--baseline-label", default="시작점")
    ap.add_argument("--arr-k", type=float, default=15.0); ap.add_argument("--arr-v", type=float, default=0.45)
    a = ap.parse_args()
    global ARR_K, ARR_V
    ARR_K, ARR_V = a.arr_k, a.arr_v
    hist_path = os.path.join(HERE, "..", "checkpoints", f"trend_{a.label}.json")
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    base = None
    if a.baseline_ckpt:
        ok_, n_, ex_ = measure(a.baseline_ckpt, a.side, a.n, a.seed0, a.level, a.workers); base = [ok_, n_, 0] + ex_
        base = [ok_, n_, 0, *ex_]
    while True:
        files = glob.glob(a.glob)
        if files:
            ck = max(files, key=key); st = key(ck)
            if not hist or hist[-1][0] != st:
                ok, n, ex_ = measure(ck, a.side, a.n, a.seed0, a.level, a.workers)
                hist.append([st, ok, n] + ex_); json.dump(hist, open(hist_path, "w"))
            trend = " → ".join(f"{h[1] / h[2]:.0%}" for h in hist[-6:])
            last = hist[-1]; pct = last[1] / last[2]
            lines = [f"[{a.label}] 누운 바닥 정책 이어서 학습", f"스텝 {last[0] / 1e6:.1f}M / {a.total_steps / 1e6:.0f}M ({last[0] / a.total_steps:.0%})",
                     f"인계 영역 도달 {last[1]}/{last[2]} = {pct:.1%} (고정 시드 {a.seed0}~, 레벨 {a.level}, 결정적 행동)", f"최근 추세: {trend}"]
            if len(last) > 3:
                lines.append(f"도착 순간: COM 전진속도 평균 {last[3]:+.2f} m/s (0.2 이상 {last[5]:.0%}), 기울기 평균 {last[4]:.1f}°, 도착 시각 중앙값 {last[6]:.1f}초, 도착 점수 평균 {last[7]:+.1f}")
                if base is not None:
                    lines.append(f"기준({a.baseline_label}): 속도 {base[3]:+.2f} m/s (0.2 이상 {base[5]:.0%}), 기울기 {base[4]:.1f}°, 도착 {base[6]:.1f}초, 점수 {base[7]:+.1f}, 도달 {base[1] and base[0] / base[1]:.0%}")
            if len(last) > 8:
                lines.append(f"도착 상태를 서기 정책(big1)이 한 번에 받아내는 비율 {last[8]:.0%}" + (f" (기준 {base[8]:.0%})" if base is not None and len(base) > 8 else ""))
            if a.ref is not None:
                lines.append(f"{a.ref_label} {a.ref:.1f}% (측정 오차 ±2.7%p, 체크포인트 간 흔들림 ±4%p)")
            send_message("\n".join(lines))
        if a.once or (files and key(max(files, key=key)) >= a.total_steps):
            break
        time.sleep(a.interval_minutes * 60)


if __name__ == "__main__":
    main()
