"""What makes an arrival catchable? Collects supine arrival states of the supine floor policy (deterministic, level 0.7), labels each with the stand policy's real 3 s catch (benchmark criterion), then predicts the label
from the body state (joint positions without root_x, all joint velocities: 33 numbers) with a logistic regression and a small MLP (5-fold cross-validated AUC) and ranks the variables: single-variable AUC, standardised
logistic weights, MLP permutation importance. Saves the labelled states to checkpoints/analysis/arrival_labels.npz (reusable as a learned arrival score).
Usage: .venv/bin/python -I scripts/analysis/arrival_classifier.py [--n 4000] [--seed0 410000] [--workers 8]"""
import argparse, os, sys
from multiprocessing import Pool
ROOT = "/home/ubuntu/workspace/standai/"
sys.path.insert(0, ROOT + "scripts"); sys.path.insert(0, ROOT + "scripts/analysis")
import numpy as np
from arrival_push_diag import collect, catch

JN = ["root_x", "root_z", "root_ry", "waist", "head", "shoulder_r", "elbow_r", "shoulder_l", "elbow_l", "hip_r", "knee_r", "ankle_r", "toe_r", "hip_l", "knee_l", "ankle_l", "toe_l"]
NAMES = [f"q:{n}" for n in JN[1:]] + [f"v:{n}" for n in JN]


def feats(q, v):
    return np.concatenate([q[1:17], v[0:17]])


def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.all() or (~y).all(): return float("nan")
    r = np.argsort(np.argsort(x)) + 1.0
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def fit_logistic(X, y, l2=1e-2):
    import torch
    X = torch.tensor(X, dtype=torch.float32); y = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def fit_mlp(X, y, seed=0, epochs=150):
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-3)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]; opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[idx])[:, 0], yt[idx]).backward(); opt.step()
    return net


def mlp_scores(net, X):
    import torch
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32))[:, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--seed0", type=int, default=410000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--floor", default=ROOT + "checkpoints/best_floor_supine_fdist.zip"); ap.add_argument("--stand", default=ROOT + "checkpoints/best_stand_big1.zip")
    a = ap.parse_args()
    out = ROOT + "checkpoints/analysis/arrival_labels.npz"
    if os.path.exists(out):
        d = np.load(out); Q, V, y = d["q"], d["v"], d["y"]
    else:
        seeds = list(range(a.seed0, a.seed0 + a.n)); ch = [seeds[i::a.workers] for i in range(a.workers)]
        with Pool(a.workers) as p:
            states = sorted([x for part in p.map(collect, [(a.floor, c) for c in ch]) for x in part], key=lambda t: t[0])
            print(f"{len(states)} arrivals from {a.n} supine starts", flush=True)
            y = np.array(p.map(catch, [(a.stand, q, v, 0.0, i) for i, (s, q, v) in enumerate(states)]), dtype=bool)
        Q = np.array([q for _, q, _ in states]); V = np.array([v for _, _, v in states])
        np.savez(out, q=Q, v=V, y=y)
    X = np.array([feats(q, v) for q, v in zip(Q, V)])
    print(f"{len(y)} labelled arrivals, stand policy catches {y.mean():.1%}", flush=True)
    # single-variable AUC
    ua = [(abs(auc(X[:, j], y) - 0.5) + 0.5, auc(X[:, j], y), NAMES[j]) for j in range(X.shape[1])]
    print("\nsingle-variable AUC (best 10):"); [print(f"  {n:14s} {a_:.2f}") for _, a_, n in sorted(ua, reverse=True)[:10]]
    # 5-fold CV
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); folds = np.array_split(idx, 5)
    pl = np.zeros(len(y)); pm = np.zeros(len(y)); imp = np.zeros(X.shape[1]); coefs = []
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w, b = fit_logistic(Xtr, y[tr]); pl[te] = Xte @ w + b; coefs.append(w)
        net = fit_mlp(Xtr, y[tr]); pm[te] = mlp_scores(net, Xte)
        base = auc(pm[te], y[te])
        for j in range(X.shape[1]):
            Xp = Xte.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); imp[j] += (base - auc(mlp_scores(net, Xp), y[te])) / 5
    print(f"\n5-fold CV AUC: logistic {auc(pl, y):.3f}, MLP(64x64) {auc(pm, y):.3f}")
    for name, s in (("logistic", pl), ("MLP", pm)):
        o = np.argsort(-s); n20 = len(y) // 5
        print(f"  {name}: top 20% predicted catch rate {y[o[:n20]].mean():.0%}, bottom 20% {y[o[-n20:]].mean():.0%}")
    cm = np.mean(coefs, 0)
    print("\nstandardised logistic weight (largest 10, + = more catchable):"); [print(f"  {NAMES[j]:14s} {cm[j]:+.2f}") for j in np.argsort(-np.abs(cm))[:10]]
    print("\nMLP permutation importance (AUC drop, largest 10):"); [print(f"  {NAMES[j]:14s} {imp[j]:+.3f}") for j in np.argsort(-imp)[:10]]


if __name__ == "__main__":
    main()
