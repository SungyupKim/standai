"""Why can a stand checkpoint have a LOW training fall_rate and a 0/100 gate? The log's fall_rate counts only episodes that stayed below
FALL_HEIGHT for 250 steps in a row (5 s, env.py's KO); the gate needs the torso to never go below it in 3 s.
Runs calm starts for 20 s with deterministic and with sampled actions: time of the first drop, longest continuous run below 0.7 m,
share of steps above 0.7 m, and whether the 250-step KO would have fired. Usage: stand_longrun_diag.py <ckpt> [n]"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
from env import FALL_HEIGHT
m = PPO.load(sys.argv[1], device="cpu"); n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
env = LocomotionEnv(stand_only=True); env.set_calm_frac(1.0)
for det in (True, False):
    first, longest, above, ko, end = [], [], [], 0, []
    for seed in range(n):
        obs, _ = env.reset(seed=seed); run = best = 0; f = None; up = 0
        for t in range(1000):
            a, _ = m.predict(obs, deterministic=det); obs, *_ = env.step(a)
            z = float(env.data.xpos[env.a_torso_id][2])
            if z < FALL_HEIGHT:
                run += 1; best = max(best, run)
                if f is None: f = t
            else:
                run = 0; up += 1
        first.append(f if f is not None else 1000); longest.append(best); above.append(up / 1000); ko += best >= 250; end.append(z)
    print(f"{'deterministic' if det else 'sampled     '}: first drop below 0.7 m at step median {int(np.median(first))} | longest run below 0.7 m median {int(np.median(longest))} steps | "
          f"share of 20 s above 0.7 m median {np.median(above):.0%} | 250-step KO would fire in {ko}/{n} | torso at 20 s median {np.median(end):.2f} m", flush=True)
