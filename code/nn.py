"""Neural components of the blend: 4th-place-style base network + two innovations.

base : inputs = leave-one-out target encodings (TE) of cell type and compound on uncentred TSVD components (K=64) of
       the visible labels; multiplicative N(1, sigma) noise on training encodings; MLP ensemble (E members, trained
       jointly) -> K components; loss = exact per-row MRRMSE including the out-of-subspace residual.
I1   : + analog target encoding: for row (c, d), a kernel-ridge weighted combination of cell c's reference rows
       (15 anchors + 2 controls, d excluded), with weights from the similarity of d to each reference compound in the
       OTHER T/NK cell types. Training rows use the same reference set, so train / test encodings are built alike.
I2   : magnitude-aware output: z = m * u with ||u|| = 1 and log m = MLP(h) + beta_c * log||TE_comp|| + gamma_c
       (beta, gamma per cell type).
Hidden rows (validation targets or test rows) never enter the SVD, the encodings' sums or the loss.
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

DEV = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = False
VARIANTS = {'base': (False, False), 'base_I1': (True, False), 'base_I2': (False, True), 'base_I1_I2': (True, True)}


@dataclass
class Config:
    K: int = 64; E: int = 8; epochs: int = 400; hidden: int = 512; dropout: float = 0.3
    wd: float = 1e-4; lr: float = 1e-3; sigma: float = 0.1; alpha: float = 1.0


def loo_te(Z, vis, key):
    """Mean of Z over the visible rows sharing `key`, leaving the row itself out when it is visible."""
    te = np.zeros_like(Z)
    for k in np.unique(key):
        rows = np.flatnonzero(key == k); v = rows[vis[rows]]
        S = Z[v].sum(0); n = len(v)
        for i in rows:
            if vis[i]: te[i] = (S - Z[i]) / (n - 1) if n > 1 else 0.
            else: te[i] = S / n if n else 0.
    return te


def prepare(tab, hidden, cfg):
    """TSVD of the visible rows, LOO target encodings and the analog target encoding (I1)."""
    Y = torch.tensor(tab.Y, dtype=torch.float32, device=DEV)
    N, cells, comps, row_of = tab.N, tab.cells, tab.comps, tab.row_of
    vis = np.ones(N, bool); vis[hidden] = False
    Xv = Y[torch.tensor(np.flatnonzero(vis), device=DEV)]
    _, _, Vh = torch.linalg.svd(Xv, full_matrices=False); V = Vh[:cfg.K].T.contiguous()
    Zt = Y @ V; res = (Y.square().sum(1) - Zt.square().sum(1)).clamp(min=0)
    Z = Zt.double().cpu().numpy()
    cte, dte = loo_te(Z, vis, tab.c_row), loo_te(Z, vis, tab.d_row)
    # compound profiles excluding a given cell type: mean over the OTHER visible T/NK rows of the compound
    prof = {}
    for c in range(len(cells)):
        Pc = np.zeros((len(comps), cfg.K)); cnt = np.zeros(len(comps))
        for s in tab.TNK:
            if s == c: continue
            for d in range(len(comps)):
                r = row_of.get((s, d))
                if r is not None and vis[r]: Pc[d] += Z[r]; cnt[d] += 1
        prof[c] = Pc / np.maximum(cnt, 1)[:, None]
    ate = np.zeros_like(Z)
    for i in range(N):
        c, d = tab.c_row[i], tab.d_row[i]
        ref = [row_of[(c, a)] for a in tab.REF if a != d and (c, a) in row_of and vis[row_of[(c, a)]]]
        if len(ref) < 2: ate[i] = cte[i]; continue
        Pr = prof[c][[tab.d_row[r] for r in ref]]; pm = Pr.mean(0); Pc_ = Pr - pm; q = prof[c][d] - pm
        Km = Pc_ @ Pc_.T; w = np.linalg.solve(Km + cfg.alpha * np.trace(Km) / len(ref) * np.eye(len(ref)), Pc_ @ q)
        Zr = Z[ref]; zm = Zr.mean(0); ate[i] = zm + w @ (Zr - zm)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    return dict(vis=vis, V=V, z=Zt, res=res, cte=T(cte), dte=T(dte), ate=T(ate))


class Net(torch.nn.Module):
    """E independent 2-hidden-layer MLPs evaluated as one batched model."""

    def __init__(self, E, nin, K, H, mag, zlog_med, dlog_med, n_cells):
        super().__init__()
        def P(*s, scale): return torch.nn.Parameter((torch.rand(*s, device=DEV) * 2 - 1) * scale)
        self.mag = mag
        self.W1 = P(E, nin, H, scale=nin ** -0.5); self.b1 = P(E, 1, H, scale=0.)
        self.W2 = P(E, H, H, scale=H ** -0.5); self.b2 = P(E, 1, H, scale=0.)
        if mag:
            self.Wu = P(E, H, K, scale=H ** -0.5); self.bu = P(E, 1, K, scale=0.)
            self.Wm = P(E, H, 1, scale=0.01 * H ** -0.5)
            self.bm = torch.nn.Parameter(torch.full((E, 1, 1), float(zlog_med - dlog_med), device=DEV))
            self.beta = torch.nn.Parameter(torch.ones(E, n_cells, device=DEV))
            self.gam = torch.nn.Parameter(torch.zeros(E, n_cells, device=DEV))
        else:
            self.W3 = P(E, H, K, scale=0.01 * H ** -0.5); self.b3 = P(E, 1, K, scale=0.)

    def forward(self, x, ci, dlog, s, drop, zmax):
        h = F.gelu(x @ self.W1 + self.b1); h = F.dropout(h, drop, self.training)
        h = F.gelu(h @ self.W2 + self.b2); h = F.dropout(h, drop, self.training)
        if not self.mag:
            return (h @ self.W3 + self.b3) * s
        u = h @ self.Wu + self.bu; u = u / (u.norm(dim=-1, keepdim=True) + 1e-6)
        logm = (h @ self.Wm + self.bm).squeeze(-1) + self.beta[:, ci] * dlog + self.gam[:, ci]
        return torch.exp(logm.clamp(max=zmax)).unsqueeze(-1) * u


def fit_predict(S, tab, qry, variant, seed, cfg):
    """Train one ensemble on the visible rows; return its median prediction for `qry` in TSVD coordinates."""
    use_ate, mag = VARIANTS[variant]
    tr = torch.tensor(np.flatnonzero(S['vis']), device=DEV); q = torch.tensor(qry, device=DEV)
    feats = [S['cte'], S['dte']] + ([S['ate']] if use_ate else [])
    X = torch.cat(feats, 1); fs = X[tr].std(0) + 1e-6
    ci_all = torch.tensor(tab.c_row, device=DEV)
    zt, rt = S['z'][tr], S['res'][tr]; s = zt.std(0) + 1e-3
    zlog = torch.log(zt.norm(dim=1) + 1e-6); zmax = float(zlog.max()) + 1.0
    dnorm = torch.log(S['dte'].norm(dim=1) + 1e-3)
    torch.manual_seed(seed)
    m = Net(cfg.E, X.shape[1], cfg.K, cfg.hidden, mag, float(zlog.median()), float(dnorm[tr].median()), len(tab.cells))
    dec = [p_ for n, p_ in m.named_parameters() if n.startswith('W')]
    rest = [p_ for n, p_ in m.named_parameters() if not n.startswith('W')]
    opt = torch.optim.AdamW([{'params': dec, 'weight_decay': cfg.wd}, {'params': rest, 'weight_decay': 0.}], lr=cfg.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    Xtr, citr = X[tr], ci_all[tr]
    m.train()
    for _ in range(cfg.epochs):
        noise = 1 + cfg.sigma * torch.randn(cfg.E, *Xtr.shape, device=DEV)      # LOO-encoder style multiplicative noise
        Xn = Xtr.unsqueeze(0) * noise
        dl = torch.log((S['dte'][tr].unsqueeze(0) * noise[..., cfg.K:2 * cfg.K]).norm(dim=-1) + 1e-3)
        out = m(Xn / fs, citr, dl, s, cfg.dropout, zmax)
        loss = torch.sqrt(((out - zt).square().sum(-1) + rt) / tab.G + 1e-8).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        Xq = (X[q] / fs).unsqueeze(0).expand(cfg.E, -1, -1)
        zq = m(Xq, ci_all[q], dnorm[q].unsqueeze(0).expand(cfg.E, -1), s, 0., zmax).median(0).values
    return zq


def ensemble_predict(S, tab, qry, variant, seeds, base_seed, cfg):
    """Median over `seeds` ensembles (seed = base_seed + 1000 r), mapped back to gene space.
    Returns (prediction [len(qry), G] float64, per-seed TSVD coordinates)."""
    zs = torch.stack([fit_predict(S, tab, qry, variant, base_seed + 1000 * r, cfg) for r in range(seeds)])
    return (zs.median(0).values @ S['V'].T).double().cpu().numpy(), zs
