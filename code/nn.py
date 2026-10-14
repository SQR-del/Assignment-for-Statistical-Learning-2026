"""Torch model zoo for the perturbation benchmark.

Low-level, fully parameterised trainers (all keyword-tunable) plus the network
definitions.  Higher-level method wrappers that operate on a fold *context* live
in the per-method `run.py` files under `methods/`.

Conventions
-----------
* Targets are compressed with a low-rank reduction (TruncatedSVD on raw targets
  or PCA on per-gene standardised targets) and the network predicts that latent;
  `inverse_transform` maps back to the 18,211-dim gene space.
* The default training loss is the row-wise RMSE (== the official metric).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD, PCA

SEED = 20251001


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def row_rmse_loss(p, z):
    return torch.mean(torch.sqrt(torch.mean((p - z) ** 2, dim=1) + 1e-8))


def make_loss(kind="rowrmse", huber_delta=1.0):
    if kind == "rowrmse":
        return row_rmse_loss
    if kind == "mse":
        return nn.functional.mse_loss
    if kind == "huber":
        return lambda p, z: nn.functional.smooth_l1_loss(p, z, beta=huber_delta)
    if kind == "logcosh":
        def _lc(p, z):
            d = p - z
            return torch.mean(torch.abs(d) + nn.functional.softplus(-2 * torch.abs(d)) - np.log(2.0))
        return _lc
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# low-rank target reductions
# --------------------------------------------------------------------------- #
class Reducer:
    """Invertible low-rank target reduction: 'svd' (raw) or 'pca_std' (standardised)."""

    def __init__(self, kind="svd", n_dim=25, seed=SEED):
        self.kind, self.n_dim, self.seed = kind, n_dim, seed

    def fit(self, Y):
        self.n_dim = int(min(self.n_dim, Y.shape[0] - 1, Y.shape[1]))
        if self.kind == "svd":
            self.red = TruncatedSVD(n_components=self.n_dim, random_state=self.seed).fit(Y)
        elif self.kind == "pca_std":
            self.mu = Y.mean(0); self.sd = Y.std(0); self.sd[self.sd < 1e-8] = 1.0
            self.red = PCA(n_components=self.n_dim, random_state=self.seed).fit((Y - self.mu) / self.sd)
        else:
            raise ValueError(self.kind)
        return self

    def transform(self, Y):
        if self.kind == "svd":
            return self.red.transform(Y).astype(np.float32)
        return self.red.transform((Y - self.mu) / self.sd).astype(np.float32)

    def inverse_transform(self, Z):
        if self.kind == "svd":
            return self.red.inverse_transform(Z)
        return self.red.inverse_transform(Z) * self.sd + self.mu


# --------------------------------------------------------------------------- #
# embedding MLP: (cell_id, drug_id) [+ numeric features] -> latent
# --------------------------------------------------------------------------- #
class EmbMLP(nn.Module):
    """ids-only embedding MLP."""

    def __init__(self, n_cells, n_drugs, emb=128, hidden=(256, 512, 1024), out=25, drop=0.2):
        super().__init__()
        self.ec = nn.Embedding(n_cells, emb)
        self.ed = nn.Embedding(n_drugs, emb)
        layers, prev = [], 2 * emb
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            prev = h
        layers += [nn.Linear(prev, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(torch.cat([self.ec(x[:, 0]), self.ed(x[:, 1])], -1))


class EmbMLP2(nn.Module):
    """(cell_id, drug_id) embeddings + optional numeric features -> latent."""

    def __init__(self, n_cells, n_drugs, emb, extra, hidden, out, drop=0.2):
        super().__init__()
        self.ec = nn.Embedding(n_cells, emb)
        self.ed = nn.Embedding(n_drugs, emb)
        layers, prev = [], 2 * emb + extra
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            prev = h
        layers += [nn.Linear(prev, out)]
        self.num = (nn.Sequential(nn.BatchNorm1d(extra), nn.Linear(extra, extra), nn.ReLU())
                    if extra > 0 else None)
        self.net = nn.Sequential(*layers)

    def forward(self, xc, xn):
        e = torch.cat([self.ec(xc[:, 0]), self.ed(xc[:, 1])], -1)
        if self.num is not None:
            e = torch.cat([e, self.num(xn)], -1)
        return self.net(e)


def _train_torch(net, forward, opt_params_x, epochs, bs, lr, wd, loss_fn, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(opt_params_x, lr=lr, weight_decay=wd)
    n = forward["n"]
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(*forward["call"](idx)).backward()
            opt.step()
    net.eval()


def train_embed(x_tr, y_tr, x_te, n_cells, n_drugs, emb=128, hidden=(256, 512, 1024),
                n_dim=25, epochs=400, bs=64, lr=3e-3, wd=1e-4, drop=0.2,
                reduction="svd", loss="rowrmse", huber_delta=1.0, sample_w=None, seed=SEED):
    """Train an ids-only embedding MLP and predict `x_te` in gene space."""
    torch.manual_seed(seed)
    red = Reducer(reduction, n_dim, seed).fit(y_tr)
    z = red.transform(y_tr)
    net = EmbMLP(n_cells, n_drugs, emb, tuple(hidden), z.shape[1], drop)
    Xtr = torch.tensor(x_tr, dtype=torch.long); Ztr = torch.tensor(z)
    Xte = torch.tensor(x_te, dtype=torch.long)
    loss_fn = make_loss(loss, huber_delta)
    w = None if sample_w is None else torch.tensor(sample_w, dtype=torch.float32).reshape(-1, 1)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    n = len(Xtr); net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            if w is None:
                l = loss_fn(net(Xtr[idx]), Ztr[idx])
            else:
                err = torch.sqrt(torch.mean((net(Xtr[idx]) - Ztr[idx]) ** 2, dim=1) + 1e-8)
                l = torch.mean(err * w[idx, 0])
            l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return red.inverse_transform(net(Xte).numpy()).astype(np.float32)


def train_hybrid(xc_tr, xn_tr, y_tr, xc_te, xn_te, n_cells, n_drugs, emb=128,
                 hidden=(256, 512, 1024), n_dim=25, epochs=400, bs=64, lr=3e-3, wd=1e-4,
                 drop=0.2, reduction="svd", loss="rowrmse", seed=SEED):
    """Embedding MLP with extra numeric features (e.g. PCA of dsig / baseline)."""
    torch.manual_seed(seed)
    red = Reducer(reduction, n_dim, seed).fit(y_tr)
    z = red.transform(y_tr)
    extra = 0 if xn_tr is None else xn_tr.shape[1]
    net = EmbMLP2(n_cells, n_drugs, emb, extra, tuple(hidden), z.shape[1], drop)
    Xc = torch.tensor(xc_tr, dtype=torch.long); Z = torch.tensor(z)
    Xn = None if extra == 0 else torch.tensor(xn_tr, dtype=torch.float32)
    Xc_te = torch.tensor(xc_te, dtype=torch.long)
    Xn_te = None if extra == 0 else torch.tensor(xn_te, dtype=torch.float32)
    loss_fn = make_loss(loss)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    n = len(Xc); net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(net(Xc[idx], None if Xn is None else Xn[idx]), Z[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return red.inverse_transform(net(Xc_te, Xn_te).numpy()).astype(np.float32)


def multi_seed_median(fn, reps, base_seed, **kw):
    """Median over `reps` seeds of a predictor fn(**kw, seed=base_seed+r)."""
    ps = [fn(seed=base_seed + r, **kw) for r in range(reps)]
    return np.median(np.stack(ps), 0)


# --------------------------------------------------------------------------- #
# SMILES character CNN molecular encoder
# --------------------------------------------------------------------------- #
def build_smiles(df, max_len=120):
    uniq = df[["drug", "SMILES"]].drop_duplicates("drug")
    chars = sorted(set("".join(uniq.SMILES.fillna("").astype(str))))
    c2i = {c: i + 1 for i, c in enumerate(chars)}   # 0 = pad
    d2s = {}
    for d, s in zip(uniq.drug, uniq.SMILES):
        ids = [c2i.get(c, 0) for c in str(s)][:max_len]
        ids += [0] * (max_len - len(ids))
        d2s[d] = np.asarray(ids, dtype=np.int64)
    return d2s, len(chars) + 1


class SmilesCNN(nn.Module):
    def __init__(self, vocab, n_cells, n_drugs, char_dim=16, conv=64, k=5,
                 use_drug_emb=True, emb=128, hidden=(256, 512, 1024), out=25, drop=0.2):
        super().__init__()
        self.ce = nn.Embedding(vocab, char_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_dim, conv, k, padding=k // 2)
        self.ec = nn.Embedding(n_cells, emb)
        self.use_drug = use_drug_emb
        if use_drug_emb:
            self.ed = nn.Embedding(n_drugs, emb)
        prev = conv + emb + (emb if use_drug_emb else 0)
        layers = []
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            prev = h
        layers += [nn.Linear(prev, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, sym, xc):
        e = self.ce(sym).transpose(1, 2)
        z = torch.relu(self.conv(e)).max(dim=2).values
        parts = [z, self.ec(xc[:, 0])]
        if self.use_drug:
            parts.append(self.ed(xc[:, 1]))
        return self.net(torch.cat(parts, -1))


def train_smiles(x_tr, sym_tr, y_tr, x_te, sym_te, vocab, n_cells, n_drugs, n_dim=25,
                 use_drug_emb=True, epochs=400, bs=64, lr=3e-3, wd=1e-4, drop=0.2, seed=SEED):
    torch.manual_seed(seed)
    red = Reducer("svd", n_dim, seed).fit(y_tr)
    z = red.transform(y_tr)
    net = SmilesCNN(vocab, n_cells, n_drugs, use_drug_emb=use_drug_emb, drop=drop, out=z.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    S = torch.tensor(sym_tr, dtype=torch.long); X = torch.tensor(x_tr, dtype=torch.long)
    Z = torch.tensor(z)
    Se = torch.tensor(sym_te, dtype=torch.long); Xe = torch.tensor(x_te, dtype=torch.long)
    n = len(X); net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            row_rmse_loss(net(S[idx], X[idx]), Z[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return red.inverse_transform(net(Se, Xe).numpy()).astype(np.float32)


# --------------------------------------------------------------------------- #
# nonlinear target autoencoder (JAX-notebook idea, torch re-implementation)
# --------------------------------------------------------------------------- #
class AE(nn.Module):
    def __init__(self, g, hidden, k, drop=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(g, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, k))
        self.dec = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, g))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def train_ae(Y, k, hidden=256, epochs=150, bs=64, lr=1e-3, wd=1e-5, seed=0):
    """Fit an autoencoder on target vectors; return (net, latent codes)."""
    torch.manual_seed(seed)
    net = AE(Y.shape[1], hidden, k)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Yt = torch.tensor(Y, dtype=torch.float32)
    n = Yt.shape[0]; net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            rec, _ = net(Yt[idx])
            nn.functional.mse_loss(rec, Yt[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        z = net.enc(Yt).numpy().astype(np.float32)
    return net, z


def train_ae_target(x_tr, y_tr, x_te, n_cells, n_drugs, k=25, hidden=256, epochs=150,
                    embed_epochs=400, standardize=False, seed=SEED):
    """Autoencoder low-rank target + embedding MLP on its latent (decoded)."""
    torch.manual_seed(seed)
    mu = sd = None
    Y = y_tr
    if standardize:
        mu, sd = y_tr.mean(0), y_tr.std(0); sd[sd < 1e-8] = 1.0
        Y = (y_tr - mu) / sd
    net, z = train_ae(Y, k, hidden=hidden, epochs=epochs, seed=seed)
    emb = EmbMLP(n_cells, n_drugs, 64, (256, 1024), z.shape[1], 0.2)
    opt = torch.optim.Adam(emb.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.tensor(x_tr, dtype=torch.long); Ztr = torch.tensor(z)
    Xte = torch.tensor(x_te, dtype=torch.long)
    n = len(Xtr); emb.train()
    for _ in range(embed_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            opt.zero_grad(); row_rmse_loss(emb(Xtr[idx]), Ztr[idx]).backward(); opt.step()
    emb.eval()
    with torch.no_grad():
        zhat = emb(Xte)
        yp = net.dec(zhat).numpy()
    if standardize:
        yp = yp * sd + mu
    return yp.astype(np.float32)


# --------------------------------------------------------------------------- #
# full-output MLP (predicts all 18,211 genes directly from compact features)
# --------------------------------------------------------------------------- #
def train_full_mlp(Xtr, Xte, y_tr, hidden=(512, 256), epochs=150, bs=64, lr=1e-3, wd=1e-4,
                   in_drop=0.0, dropout=0.1, cosine=True, residual=False, dsig_tr=None,
                   dsig_te=None, seed=SEED):
    """Standardised-target MLP for the full gene space.  If `residual`, the net
    predicts y - dsig and dsig is added back."""
    from models import _std_targets
    torch.manual_seed(seed)
    T = (y_tr - dsig_tr) if residual else y_tr
    mu, sd = _std_targets(T)
    Ztr = ((T - mu) / sd).astype(np.float32)
    layers, prev = [], Xtr.shape[1]
    if in_drop:
        layers += [nn.Dropout(in_drop)]
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, Ztr.shape[1])]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); Ztr_t = torch.tensor(Ztr)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    n = len(Xtr_t); net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            nn.functional.mse_loss(net(Xtr_t[idx]), Ztr_t[idx]).backward(); opt.step()
        if sched is not None:
            sched.step()
    net.eval()
    with torch.no_grad():
        pred = (net(Xte_t).numpy() * sd + mu).astype(np.float32)
    return (pred + dsig_te.astype(np.float32)) if residual else pred


# --------------------------------------------------------------------------- #
# CP / bilinear tensor factorisation of the (cell x drug x gene) response
# --------------------------------------------------------------------------- #
def fit_cp(cells_tr, drugs_tr, Y_tr, n_cells, n_drugs, R=20, steps=1200, lr=0.01, wd=1e-5, seed=0):
    """Y[c,d,g] ~ mu[g] + A[c,g] + B[d,g] + sum_k U[c,k] W[d,k] Z[k,g]."""
    torch.manual_seed(seed)
    G = Y_tr.shape[1]
    c_tr = torch.tensor(cells_tr, dtype=torch.long)
    d_tr = torch.tensor(drugs_tr, dtype=torch.long)
    Yt = torch.tensor(Y_tr, dtype=torch.float32)
    mu = torch.tensor(Y_tr.mean(0), dtype=torch.float32)
    A = torch.zeros(n_cells, G, requires_grad=True)
    B = torch.zeros(n_drugs, G)
    for dd in range(n_drugs):
        m = d_tr == dd
        if m.any():
            B[dd] = (Yt[m] - mu).mean(0)
    B = B.clone().requires_grad_(True)
    U = (torch.randn(n_cells, R) * 0.01).requires_grad_(True)
    W = (torch.randn(n_drugs, R) * 0.1).requires_grad_(True)
    Z = (torch.randn(R, G) * 0.01).requires_grad_(True)
    opt = torch.optim.Adam([A, B, U, W, Z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = mu + A[c_tr] + B[d_tr] + (U[c_tr] * W[d_tr]) @ Z
        loss = row_rmse_loss(pred, Yt)
        reg = wd * (A.square().mean() + B.square().mean() + U.square().mean()
                    + W.square().mean() + Z.square().mean())
        (loss + reg).backward(); opt.step()
    return {k: v.detach() for k, v in dict(mu=mu, A=A, B=B, U=U, W=W, Z=Z).items()}


def predict_cp(model, cells_te, drugs_te):
    c_te = torch.tensor(cells_te, dtype=torch.long); d_te = torch.tensor(drugs_te, dtype=torch.long)
    mu, A, B, U, W, Z = (model[k] for k in ["mu", "A", "B", "U", "W", "Z"])
    return (mu + A[c_te] + B[d_te] + (U[c_te] * W[d_te]) @ Z).numpy().astype(np.float32)
