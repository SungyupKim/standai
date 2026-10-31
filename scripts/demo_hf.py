"""Quick start: download the single get-up network from Hugging Face and try it (no checkpoints needed in git).

    python scripts/demo_hf.py                 # 50 supine + 50 prone starts (~1.5 min), prints how many are standing 20 s later
    python scripts/demo_hf.py --video         # also renders a short demo video (needs pillow + ffmpeg), written to ./demo_hf.mp4 / .gif
    python scripts/demo_hf.py --file getup_e2e_256.zip --n 50 --seed0 98000

Model: https://huggingface.co/SungyupKim/standai-getup (default getup_e2e_512.zip, the 512x512 network after end-to-end PPO).
"standing" = torso >= 0.95 m and |pitch| < 30 deg for the last 25 of 1000 steps (20 s), the criterion used everywhere in this repo.
Starts the environment could not make lie down (1.2% of prone starts) are skipped."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SungyupKim/standai-getup")
    ap.add_argument("--file", default="getup_e2e_512.zip")
    ap.add_argument("--n", type=int, default=50, help="starts per side (small samples are noisy: 8 starts can easily show 62% for a 90% policy)")
    ap.add_argument("--seed0", type=int, default=98000)
    ap.add_argument("--video", action="store_true", help="also render a short demo video (pillow + ffmpeg)")
    ap.add_argument("--out", default="demo_hf", help="video basename (--video)")
    a = ap.parse_args()
    for k in list(os.environ):          # training variables change the start distribution
        if k.startswith("STANDAI_"):
            del os.environ[k]
    import numpy as np
    from huggingface_hub import hf_hub_download
    from stable_baselines3 import PPO
    from getup_env import GetUpEnv

    path = hf_hub_download(a.repo, a.file)
    model = PPO.load(path, device="cpu")
    print(f"model {a.repo}/{a.file}: {sum(p.numel() for p in model.policy.parameters()):,} parameters, observation {model.observation_space.shape}, action {model.action_space.shape}")
    for side in ("supine", "prone"):
        env = GetUpEnv(horizon=1000, side=side)
        ok = tried = 0
        seed = a.seed0
        while tried < a.n:
            obs, _ = env.reset(seed=seed); seed += 1
            if float(env.data.xpos[env.a_torso_id][2]) >= 0.5:
                continue                      # not a lying start
            standing = []
            for _ in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _, info = env.step(action)
                standing.append(info["standing"])
            ok += all(standing[-25:]); tried += 1
        print(f"{side:6s} start: standing 20 s later in {ok}/{tried} ({ok / tried:.0%}) of seeds {a.seed0}..{seed - 1}")
    if a.video:
        subprocess.run([sys.executable, os.path.join(HERE, "render_demo.py"), "--student", path, "--supine-ok", "1", "--prone-ok", "1",
                        "--seed0", str(a.seed0), "--seconds", "10", "--out", a.out, "--caption", "Single network from Hugging Face (simulation)."], check=True)
        print(f"wrote {a.out}.mp4 and {a.out}.gif")


if __name__ == "__main__":
    main()
