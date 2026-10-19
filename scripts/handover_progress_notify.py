"""KakaoTalk progress pings for a hand-over policy run (train_handover.py): training success / fail rate, ladder level and how far ahead of the front foot the
COM may be (from the log), plus every interval the deterministic hand-over -> stand chain on the eval states (analysis/handover_eval.py first line).
Usage: handover_progress_notify.py --log ../checkpoints/handover_train_h3.log --ckpt-glob '../checkpoints/autosave/ppo_handover_h3_*_steps.zip' [--side prone] [--interval-minutes 30]"""
import argparse, glob, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kakao_notify import send_message


def last(log, key):
    m = None
    for line in open(log, errors="ignore"):
        if re.match(rf"\|\s+{re.escape(key)}\s+\|", line):
            m = line
    return float(m.split("|")[2]) if m else None


def ladder_line(log):
    m = None
    for line in open(log, errors="ignore"):
        if "ladder at" in line:
            m = line
    return m.split("ladder at ", 1)[1].strip() if m else "-"


def finished(log):
    return any("saved model to" in l for l in open(log, errors="ignore"))


def chain(ckpt, side, eval_set):
    py = os.path.join(HERE, "..", ".venv", "bin", "python")
    env = dict(os.environ, STANDAI_NO_RECOVERY_GATE="1")
    try:
        out = subprocess.run([py, "-I", os.path.join(HERE, "analysis", "handover_eval.py"), ckpt, "--side", side, "--set", eval_set],
                             capture_output=True, text=True, timeout=1500, env=env).stdout
        return out.strip().splitlines()[0]
    except Exception as e:
        return f"(chain eval failed: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True); ap.add_argument("--ckpt-glob", required=True)
    ap.add_argument("--side", default="prone"); ap.add_argument("--set", default="eval2_all")
    ap.add_argument("--interval-minutes", type=float, default=30.0)
    a = ap.parse_args()
    while True:
        done = finished(a.log)
        ck = max(glob.glob(a.ckpt_glob), key=os.path.getmtime, default=None)
        steps, succ, fail = last(a.log, "total_timesteps"), last(a.log, "success_rate"), last(a.log, "fail_rate")
        fmt = lambda v, f: "-" if v is None else format(v, f)
        msg = (f"[hand-over {a.side}] {'FINISHED ' if done else ''}steps {fmt(steps and int(steps), ',')}  train success {fmt(succ, '.0%')} / fail {fmt(fail, '.0%')}\n"
               f"ladder: {ladder_line(a.log)}\n")
        if ck:
            msg += "chain: " + chain(ck, a.side, a.set)
        try:
            send_message(msg)
            print(time.strftime("%H:%M"), "sent:", msg.replace("\n", " | "), flush=True)
        except Exception as e:
            print("send failed:", e, flush=True)
        if done:
            break
        time.sleep(a.interval_minutes * 60)


if __name__ == "__main__":
    main()
