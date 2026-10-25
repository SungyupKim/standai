"""Trains the final arrival-catchability classifier (MLP 64x64 on 33 state variables, labels from analysis/arrival_labels.npz) on ALL labelled arrivals and saves checkpoints/arrival_clf.pt
(state dicts of 3 seeds + feature mean/std) for the opt-in controller gate STANDAI_CLF_GATE."""
import os, sys
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts"); sys.path.insert(0, ROOT + "scripts/analysis")
import numpy as np, torch
from arrival_classifier import feats, fit_mlp, mlp_scores, auc

d = np.load(ROOT + "checkpoints/analysis/arrival_labels.npz")
X = np.array([feats(q, v) for q, v in zip(d["q"], d["v"])]); y = d["y"]
mu, sd = X.mean(0), X.std(0) + 1e-6
nets = [fit_mlp((X - mu) / sd, y, seed=s) for s in range(3)]
p = np.mean([mlp_scores(n, (X - mu) / sd) for n in nets], 0)
print(f"{len(y)} arrivals, catch {y.mean():.1%}, training AUC {auc(p, y):.3f}")
torch.save({"states": [n.state_dict() for n in nets], "mu": torch.tensor(mu, dtype=torch.float32), "sd": torch.tensor(sd, dtype=torch.float32)}, ROOT + "checkpoints/arrival_clf.pt")
