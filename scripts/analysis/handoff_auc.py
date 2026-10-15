"""Which extra state variables at hand-off separate 'stand policy holds 3 s' from 'falls'? AUC per feature (0.5 = no information).
Usage: handoff_auc.py <floor_ckpt> [n_seeds]"""
import sys; sys.path.insert(0, "/home/ubuntu/workspace/standai/scripts")
import numpy as np, mujoco
from stable_baselines3 import PPO
from floor_env import FloorEnv
from env import FALL_HEIGHT
ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
STAND = "/home/ubuntu/workspace/standai/checkpoints/autosave/ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip"
model, stand = PPO.load(ck, device="cpu"), PPO.load(STAND, device="cpu")
env = FloorEnv(standready=True, crouch_frac=0.0, max_steps=450, prone_frac=1.0, down_timeout=False)
m = env.model
J = lambda name: m.joint(name).qposadr[0]
V = lambda name: m.joint(name).dofadr[0]
root = int(m.body_rootid[env.a_torso_id])
def feats(d):
    foot_x = [float(d.site_xpos[s][0]) for s in env.a_feet]
    com_x = float(d.subtree_com[root][0])
    limb_v = np.array([d.qvel[V(n_)] for n_ in ("a_waist", "a_head", "a_shoulder_r", "a_elbow_r", "a_shoulder_l", "a_elbow_l",
                                               "a_hip_r", "a_knee_r", "a_ankle_r", "a_hip_l", "a_knee_l", "a_ankle_l")])
    foot_down, _ = env._getup_contacts()
    return {
        "COM x - mean foot x": com_x - float(np.mean(foot_x)),
        "COM x - front foot x": com_x - max(foot_x),
        "COM x - rear foot x": com_x - min(foot_x),
        "|foot x gap|": abs(foot_x[0] - foot_x[1]),
        "root vx": float(d.qvel[V("a_root_x")]),
        "|root vx|": abs(float(d.qvel[V("a_root_x")])),
        "limb joint speed norm": float(np.linalg.norm(limb_v)),
        "|hip speed| mean": float((abs(d.qvel[V("a_hip_r")]) + abs(d.qvel[V("a_hip_l")])) / 2),
        "|knee speed| mean": float((abs(d.qvel[V("a_knee_r")]) + abs(d.qvel[V("a_knee_l")])) / 2),
        "|waist speed|": abs(float(d.qvel[V("a_waist")])),
        "both knees: max bend": float(max(-d.qpos[J("a_knee_r")], -d.qpos[J("a_knee_l")]) * 57.3),
        "hip angle mean": float((d.qpos[J("a_hip_r")] + d.qpos[J("a_hip_l")]) / 2 * 57.3),
        "ankle angle mean": float((d.qpos[J("a_ankle_r")] + d.qpos[J("a_ankle_l")]) / 2 * 57.3),
        "shoulder angle mean": float((d.qpos[J("a_shoulder_r")] + d.qpos[J("a_shoulder_l")]) / 2 * 57.3),
        "elbow angle mean": float((d.qpos[J("a_elbow_r")] + d.qpos[J("a_elbow_l")]) / 2 * 57.3),
        "waist angle": float(d.qpos[J("a_waist")] * 57.3),
        "root pitch": float(((d.qpos[env.a_ry_qpos] + np.pi) % (2*np.pi) - np.pi) * 57.3),
        "pitch rate": float(d.qvel[env._ry_dof] * 57.3),
        "vz": float(d.qvel[env.a_z_dof]),
    }
rows, held = [], []
for seed in range(n):
    obs, _ = env.reset(seed=seed)
    for t in range(450):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        if info.get("floor_success"):
            f = feats(env.data); fell = False
            for _ in range(150):
                a, _ = stand.predict(obs, deterministic=True); obs, *_ = env.step(a)
                if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT: fell = True; break
            rows.append(f); held.append(not fell); break
        if term or trunc: break
held = np.array(held); keys = list(rows[0])
print(f"{ck.split('_')[-2]} steps: reached {len(rows)}/{n}, held {held.sum()} ({held.mean():.0%})")
def auc(x, y):
    order = np.argsort(x); ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):                      # average ranks for ties
        idx = x == v
        if idx.sum() > 1: ranks[idx] = ranks[idx].mean()
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
res = sorted(((abs(auc(np.array([r[k] for r in rows]), held) - 0.5), k) for k in keys), reverse=True)
print(f"{'feature':>26} | AUC (>0.5: larger value -> held)  | held mean / not-held mean")
for _, k in res:
    x = np.array([r[k] for r in rows])
    print(f"{k:>26} | {auc(x, held):.2f}                            | {x[held].mean():+8.2f} / {x[~held].mean():+8.2f}")
