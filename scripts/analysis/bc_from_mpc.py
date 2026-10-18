"""Behaviour cloning of the stand policy on MPC demonstrations (checkpoints/mpc_demos_*.npz from mpc_demos.py).
Regresses the policy's deterministic action (the mean) on MPC's 12 joint commands (action[0], the horizontal thrust, is zeroed by the env and not trained);
log_std and the critic are left as they are. Validation = held-out demo STATES (not time steps). Writes a normal PPO zip you can benchmark.
Usage: .venv/bin/python -I scripts/analysis/bc_from_mpc.py --init CKPT --demos checkpoints/mpc_demos_a.npz --out checkpoints/bc_v37_a.zip [--epochs 60] [--lr 1e-4]"""
import argparse
import sys

ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts")
import numpy as np
import torch
from stable_baselines3 import PPO

ap = argparse.ArgumentParser()
ap.add_argument("--init", default=ROOT + "checkpoints/autosave/ppo_stand_v37_ladder_20261011_212607_96000000_steps.zip")
ap.add_argument("--demos", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--val-frac", type=float, default=0.1)
a = ap.parse_args()
torch.set_num_threads(2)

d = np.load(a.demos)
obs, acts, state = d["obs"], d["acts"], d["state"]
ids = np.unique(state)
rng = np.random.default_rng(0)
val_ids = set(rng.choice(ids, max(1, int(len(ids) * a.val_frac)), replace=False).tolist())
va = np.array([s in val_ids for s in state])
print(f"demos: {len(ids)} states, {len(obs)} steps ({va.sum()} validation steps from {len(val_ids)} held-out states)", flush=True)

model = PPO.load(a.init, device="cpu")
pol = model.policy
X = torch.tensor(obs, dtype=torch.float32)
Y = torch.tensor(acts, dtype=torch.float32)


def mean_action(x):
    return pol.get_distribution(x).distribution.mean[:, 1:]


def bc_loss(m, y):
    """The env clips actions to [-1, 1], so a policy mean of 3 acts like 1: saturated targets (|y| ~ 1, MPC is bang-bang a lot) only
    penalise the mean for falling short of them, interior targets are plain squared error."""
    sat = y.abs() >= 0.99
    short = torch.relu(0.99 - torch.sign(y) * m)
    return torch.where(sat, short, (m - y)) .pow(2).mean()


with torch.no_grad():
    print(f"before: val loss {bc_loss(mean_action(X[va]), Y[va]):.4f}  (train {bc_loss(mean_action(X[~va]), Y[~va]):.4f})", flush=True)
params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
opt = torch.optim.Adam(params, lr=a.lr)
tr_idx = np.where(~va)[0]
for ep in range(a.epochs):
    rng.shuffle(tr_idx)
    for k in range(0, len(tr_idx), a.batch):
        b = tr_idx[k:k + a.batch]
        loss = bc_loss(mean_action(X[b]), Y[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 10 == 9 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"epoch {ep + 1}: val loss {bc_loss(mean_action(X[va]), Y[va]):.4f}  train {bc_loss(mean_action(X[~va]), Y[~va]):.4f}", flush=True)
model.save(a.out)
print("saved", a.out)
