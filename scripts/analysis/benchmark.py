"""One-command benchmark of the whole get-up system (and of any one policy you swap in).

    .venv/bin/python -I scripts/analysis/benchmark.py [--supine CKPT] [--prone CKPT] [--stand CKPT] [--tag NAME] [--quick] [--workers 4]

Sections (each runs in its own worker process, fixed seeds, deterministic actions, stand policy takes over at the strict stand-ready region):
  stand_catch   stand policy on the 300 held-out strict hand-off states (checkpoints/eval_strict_all.npz): genuinely upright / torso >= 0.7 m at the end
                (v37@96M: 127/300, v54@20.3M: 141/300)
  stand_calm    stand policy from upright (calm) starts, 3 s gate, and the same split by initial knee bend (crouched starts are the weak spot)
  floor_supine  supine floor policy on supine-side starts only: reached the region / the stand policy then holds 3 s (never below 0.7 m)
  floor_prone   same for the prone policy on prone-side starts
  controller    rule-based GetUpController on the natural start mix, 20 s: standing at 10 s / at the end (torso >= 0.95 m, |pitch| < 30 deg); hand-over region --handover-level (default 0.7)
Prints one table and writes checkpoints/analysis/benchmark_<tag>.json. --quick uses a quarter of the seeds (about 1/3 of the time, noisier).
Run it with nice while a training run is going, e.g.  nice -n 10 .venv/bin/python -I scripts/analysis/benchmark.py --tag v48_x
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
DEFAULTS = dict(
    supine=ROOT + "checkpoints/autosave/ppo_floor_v12_20261013_204028_101475000_steps.zip",
    prone=ROOT + "checkpoints/best_prone8.zip",
    stand=ROOT + "checkpoints/best_stand_v54.zip",   # v54 @20.3M (copy of the autosave); the old reference is ppo_stand_v37_ladder_..._96000000_steps.zip
)
EVAL_SET = ROOT + "checkpoints/eval_strict_all.npz"


def _setup():
    import torch
    torch.set_num_threads(1)


SEED0 = int(os.environ.get("STANDAI_BENCH_SEED0", "0"))   # shift the seeds of the calm / floor / controller runs (fresh seeds for confirmation); the eval npz sets are fixed


def task_stand_catch(stand, q):
    _setup()
    from stand_progress_notify import handoff_catch
    up, end, n = handoff_catch(stand, EVAL_SET)
    su, _, sn = handoff_catch(stand, ROOT + "checkpoints/eval_supine_strict.npz")
    pu, _, pn = handoff_catch(stand, ROOT + "checkpoints/eval_prone_strict.npz")
    return dict(upright=up, torso_ok_at_end=end, n=n, supine=[su, sn], prone=[pu, pn])


def task_stand_calm(stand, n):
    _setup()
    import numpy as np
    from stable_baselines3 import PPO
    from env import FALL_HEIGHT
    from locomotion_env import LocomotionEnv
    m = PPO.load(stand, device="cpu")
    env = LocomotionEnv(stand_only=True)
    env.set_calm_frac(1.0)
    rows = []
    for seed in range(SEED0, SEED0 + n):
        env.reset(seed=seed)
        knee = float(np.degrees(min(env.data.qpos[env._knee_r_qpos], env.data.qpos[env._knee_l_qpos])))
        obs = env._obs()
        ok = True
        for _ in range(150):
            a, _ = m.predict(obs, deterministic=True)
            obs, *_ = env.step(a)
            if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT:
                ok = False
                break
        rows.append((knee, ok))
    out = dict(n=n, gate_first100=sum(ok for _, ok in rows[:100]), n_first100=min(n, 100), gate_all=sum(ok for _, ok in rows))
    bins = [("knee>=-10", -10, 1), ("-25..-10", -25, -10), ("-40..-25", -40, -25), ("<-40", -999, -40)]
    out["by_knee"] = {name: [sum(ok for k, ok in rows if lo <= k < hi), sum(1 for k, _ in rows if lo <= k < hi)] for name, lo, hi in bins}
    return out


def task_floor(side, floor, stand, n):
    _setup()
    from stable_baselines3 import PPO
    from env import FALL_HEIGHT
    from floor_env import FloorEnv
    from sym import load_stand
    fl, st = PPO.load(floor, device="cpu"), load_stand(stand)
    env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=450, down_timeout=False,
                   prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    reached = held = 0
    for seed in range(SEED0, SEED0 + n):
        obs, _ = env.reset(seed=seed)
        for _ in range(450):
            a, _ = fl.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            if info.get("floor_success"):
                reached += 1
                ok = True
                for _ in range(150):
                    a, _ = st.predict(obs, deterministic=True)
                    obs, *_ = env.step(a)
                    if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT:
                        ok = False
                        break
                held += ok
                break
            if term or trunc:
                break
    return dict(n=n, reached=reached, held=held)


def task_controller(supine, prone, stand, n, side=None, level=1.0, handover=None, seed0=None):
    _setup()
    import numpy as np
    from controller import GetUpController
    from floor_env import FloorEnv
    env = FloorEnv(standready=True, level=level, crouch_frac=0.0, max_steps=10 ** 6, down_timeout=False,
                   prone_frac=1.0 if side == "prone" else 0.0, supine_frac=1.0 if side == "supine" else 0.0)
    ctl = GetUpController(supine, prone, stand, fall_back=True, handover_ckpt=handover, handover_sides=(side,) if side else ("prone",))
    s10 = send = hand = 0
    T = 1000
    s0 = SEED0 if seed0 is None else seed0
    for seed in range(s0, s0 + n):
        obs, _ = env.reset(seed=seed)
        ctl.reset(env)
        hist = []
        for _ in range(T):
            obs, _, term, trunc, info = env.step(ctl.act(obs, env))
            ctl.observe(env, info)
            hist.append((float(env.data.xpos[env.a_torso_id][2]), ctl.pitch(env)))
        ok = lambda lo, hi: all(z >= 0.95 and abs(np.degrees(p)) < 30 for z, p in hist[lo:hi])
        s10 += ok(475, 500)
        send += ok(T - 25, T)
        hand += ctl.handovers > 0
    return dict(n=n, standing_10s=s10, standing_end=send, handed_over=hand)


def pct(a, b):
    return f"{a}/{b} ({a / max(b, 1):.0%})"


def main():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", default=v)
    p.add_argument("--handover", default=None, help="hand-over policy checkpoint (scripts/handover_env.py); default none")
    p.add_argument("--tag", default="default")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--handover-level", type=float, default=0.7, help="controller hand-over region: 1.0 = strict stand-ready region, lower = looser; 0.7 (default since 2026-10-18) was best with v54 (docs 3.21); pass 1.0 to reproduce older numbers")
    a = p.parse_args()
    d = 4 if a.quick else 1
    n_calm, n_floor, n_ctl = 200 // d, 200 // d, 200 // d
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {
            "stand_catch": ex.submit(task_stand_catch, a.stand, 0),
            "stand_calm": ex.submit(task_stand_calm, a.stand, n_calm),
            "floor_supine": ex.submit(task_floor, "supine", a.supine, a.stand, n_floor),
            "floor_prone": ex.submit(task_floor, "prone", a.prone, a.stand, n_floor),
            "controller": ex.submit(task_controller, a.supine, a.prone, a.stand, n_ctl, None, a.handover_level, a.handover),
            "controller_supine": ex.submit(task_controller, a.supine, a.prone, a.stand, n_ctl // 2, "supine", a.handover_level, a.handover),
            "controller_prone": ex.submit(task_controller, a.supine, a.prone, a.stand, n_ctl // 2, "prone", a.handover_level, a.handover),
        }
        res = {k: f.result() for k, f in futs.items()}
    res["checkpoints"] = {k: os.path.basename(getattr(a, k)) for k in DEFAULTS}
    res["quick"] = a.quick
    res["seconds"] = round(time.time() - t0)
    os.makedirs(ROOT + "checkpoints/analysis", exist_ok=True)
    out = ROOT + f"checkpoints/analysis/benchmark_{a.tag}.json"
    json.dump(res, open(out, "w"), indent=1)

    c, k = res["stand_catch"], res["stand_calm"]
    print(f"benchmark '{a.tag}'{' (quick)' if a.quick else ''}   supine {res['checkpoints']['supine']} | prone {res['checkpoints']['prone']} | stand {res['checkpoints']['stand']}")
    print("-" * 110)
    print(f"stand policy, strict hand-off states ({c['n']}): genuinely upright {pct(c['upright'], c['n'])}   [baseline v37@96M 127/300]")
    print(f"stand policy, upright starts: gate {pct(k['gate_first100'], k['n_first100'])} (first 100 seeds)")
    print("   by initial knee bend: " + " | ".join(f"{name} {pct(ok, tot)}" for name, (ok, tot) in k["by_knee"].items()))
    for side in ("supine", "prone"):
        f = res[f"floor_{side}"]
        print(f"floor policy, {side}-side starts ({f['n']}): reached the region {pct(f['reached'], f['n'])}, then the stand policy holds 3 s {pct(f['held'], f['n'])} of starts "
              f"({f['held'] / max(f['reached'], 1):.0%} of reached)")
    t = res["controller"]
    print(f"rule-based controller, natural start mix ({t['n']}, 20 s): standing at 10 s {pct(t['standing_10s'], t['n'])}, at the end {pct(t['standing_end'], t['n'])}, "
          f"handed over {pct(t['handed_over'], t['n'])}   [baseline 24%]")
    # start-side split: the env's natural lying-start mix is ~80% supine only because the collapse generator's hip target range is flexion-heavy
    # (docs 3.19); the real fall direction mix is unknown, so also report each side alone and their 50:50 average
    sc, pc = c["supine"], c["prone"]
    print("-" * 110)
    print("by start side (50:50 average is the number to compare across runs; the natural 80:20 mix above is a generator artefact):")
    sh, ph = res["floor_supine"]["held"] / res["floor_supine"]["n"], res["floor_prone"]["held"] / res["floor_prone"]["n"]
    cs, cp = res["controller_supine"], res["controller_prone"]
    se, pe = cs["standing_end"] / cs["n"], cp["standing_end"] / cp["n"]
    print(f"  stand catch of hand-off states : supine {sc[0]}/{sc[1]} ({sc[0] / sc[1]:.0%}) | prone {pc[0]}/{pc[1]} ({pc[0] / pc[1]:.0%}) | 50:50 {(sc[0] / sc[1] + pc[0] / pc[1]) / 2:.0%}")
    print(f"  floor + stand holds 3 s        : supine {sh:.0%} | prone {ph:.0%} | 50:50 {(sh + ph) / 2:.0%}")
    print(f"  controller, standing at 20 s   : supine {cs['standing_end']}/{cs['n']} ({se:.0%}) | prone {cp['standing_end']}/{cp['n']} ({pe:.0%}) | 50:50 {(se + pe) / 2:.0%}")
    print(f"({res['seconds']} s; saved to {out})")


if __name__ == "__main__":
    main()
