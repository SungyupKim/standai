"""Re-computes the training log's fall_rate / ep_len_mean for a stand checkpoint outside the trainer, split by start type (calm vs harsh), sampled actions
(as in training) and deterministic. a_out = the env's KO (torso below FALL_HEIGHT for 250 steps in a row, or health 0). Usage: stand_fallrate_replica.py <ckpt> [n]"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np
from stable_baselines3 import PPO
from locomotion_env import LocomotionEnv
m = PPO.load(sys.argv[1], device="cpu"); n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
env = LocomotionEnv(stand_only=True)
for calm in (1.0, 0.0):
    env.set_calm_frac(calm)
    for det in (False, True):
        lens, kos, why = [], 0, []
        for seed in range(n):
            obs, _ = env.reset(seed=1000 + seed)
            for t in range(1100):
                a, _ = m.predict(obs, deterministic=det); obs, r, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            lens.append(t + 1); kos += bool(info.get("a_out")); why.append("trunc" if trunc else "term")
        print(f"{'calm' if calm == 1.0 else 'harsh'} starts, {'deterministic' if det else 'sampled'}: a_out {kos}/{n} ({kos / n:.0%}), ep_len mean {np.mean(lens):.0f}, "
              f"ended by truncation {why.count('trunc')}/{n}", flush=True)
