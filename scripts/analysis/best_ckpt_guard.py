"""Watches a floor run's notifier log and keeps a copy of the best-evaluated checkpoint (training is never touched).
The notifier prints  [floor-notify] sent at <N> steps (alone={... 'lying': [starts, reached, held] ...})  every eval; the checkpoint
that was evaluated is the newest autosave at or below N steps. score = held + 0.5 * reached (held = the stand policy keeps it 3 s).
Usage: best_ckpt_guard.py --log <notify log> --ckpt-glob '<autosave glob with *_steps.zip>' --out <best copy path> [--interval 600]"""
import argparse, glob, os, re, shutil, time

p = argparse.ArgumentParser()
p.add_argument("--log", required=True)
p.add_argument("--ckpt-glob", required=True)
p.add_argument("--out", required=True)
p.add_argument("--interval", type=float, default=600)
a = p.parse_args()
RE = re.compile(r"sent at (\d+) steps .*'lying': \[(\d+), (\d+), (\d+)\]")
best = (-1.0, None)
while True:
    try:
        ckpts = {}
        for f in glob.glob(a.ckpt_glob):
            m = re.search(r"_(\d+)_steps\.zip$", f)
            if m and os.path.getsize(f) > 0:
                ckpts[int(m.group(1))] = f
        for line in open(a.log):
            m = RE.search(line)
            if not m or not ckpts:
                continue
            steps, n, reached, held = map(int, m.groups())
            below = [s for s in ckpts if s <= steps]
            if not below:
                continue
            s_ck = max(below)
            score = held + 0.5 * reached
            if score > best[0]:
                best = (score, s_ck)
                shutil.copy2(ckpts[s_ck], a.out + ".tmp")
                os.replace(a.out + ".tmp", a.out)
                with open(a.out + ".txt", "w") as f:
                    f.write(f"best so far: checkpoint {os.path.basename(ckpts[s_ck])}\nevaluated at {steps} steps on {n} seeds: reached {reached}, held {held}\n")
                print(f"[guard] new best: {s_ck} steps (reached {reached}/{n}, held {held})", flush=True)
    except Exception as e:
        print(f"[guard] error: {e}", flush=True)
    time.sleep(a.interval)
