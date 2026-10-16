"""Runs a stand checkpoint in a SubprocVecEnv built exactly like train_locomotion.py (make_vec_env + same env_kwargs), with optional training-time
scaffold settings, and reports completed-episode length / a_out / mean torso z. Used to find why the trainer's fall_rate disagrees with a single env.
Usage: stand_vecenv_replica.py <ckpt> <variant: plain|knee|all> [episodes]"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from locomotion_env import LocomotionEnv
if __name__ == "__main__":
    ck, variant = sys.argv[1], sys.argv[2]; need = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    venv = make_vec_env(LocomotionEnv, n_envs=4, vec_env_cls=SubprocVecEnv,
                        env_kwargs={"stand_only": True, "ground_shaping": False, "getup_reward": False, "knee_gate": False, "head_progress": 0.0, "head_record": 0.0})
    m = PPO.load(ck, env=venv, device="cpu")
    if variant in ("knee", "all"):
        venv.env_method("set_min_knee_deg", -78.0)
    if variant == "all":
        venv.env_method("set_calm_frac", 0.5); venv.env_method("set_height_reward_scale", 15.0)
    obs = venv.reset(); lens, kos = [], []
    zs = []
    while len(lens) < need:
        a, _ = m.predict(obs, deterministic=False); obs, r, dones, infos = venv.step(a)
        for i, d in enumerate(dones):
            if d:
                lens.append(infos[i].get("episode", {}).get("l", -1)); kos.append(bool(infos[i].get("a_out")))
    print(f"variant {variant}: {len(lens)} episodes, a_out {sum(kos)}/{len(kos)}, ep_len mean {np.mean(lens):.0f}", flush=True)
    venv.close()
