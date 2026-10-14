#!/usr/bin/env python3
"""Ablation: strong base (4th-place style) + two innovations, on leak-free internal protocols.

base   : inputs = leave-one-out target encodings (cell type, compound) of TSVD components (uncentred, as 4th place);
         multiplicative N(1, sigma) noise on training encodings; MLP ensemble -> K components; exact MRRMSE loss.
I1     : + analog target encoding: for row (c,d), kernel-ridge weighted combination of cell c's ANCHOR rows (15 shared
         anchors + 2 controls, d excluded), weights from similarity of d to each anchor in the OTHER T/NK cell types.
         Training rows use the same anchor-only reference set, so train/test encodings are built identically.
I2     : magnitude-aware output: z = m * u, ||u|| = 1, log m = MLP(h) + beta_c * log||TE_comp|| + gamma_c (learned per cell type).
magic  : 4th-place post-processing (shrink part only, M=1): p * min(mean|p|^0.6, 1), applied to base predictions.
Protocols: A (targets NK/CD4/CD8/Treg, hide non-control non-anchor rows, score official test compounds), A15 = A without CD8,
           V1 (B/Myeloid leave-one-anchor-out), V1w (anchors weak in the sources).
Pre-declared objective: mean(V1w / 1.1625, A15 / 0.9894)  (KRR v1 = official private 0.830).
"""
import os, sys, time, json, argparse
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--out', default='/tmp/sqr_claude_diag/abl')
p.add_argument('--K', type=int, default=64); p.add_argument('--E', type=int, default=8); p.add_argument('--seeds', type=int, default=3)
p.add_argument('--epochs', type=int, default=400); p.add_argument('--hidden', type=int, default=512)
p.add_argument('--dropout', type=float, default=0.3); p.add_argument('--wd', type=float, default=1e-4)
p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--sigma', type=float, default=0.1)
p.add_argument('--alpha', type=float, default=1.0); p.add_argument('--seed', type=int, default=20260930)
p.add_argument('--variants', default='base,base_I1,base_I2,base_I1_I2')
p.add_argument('--smoke', action='store_true')
p.add_argument('--final', default='', help='variant to train on ALL rows and export in Kaggle id order')
p.add_argument('--poison_check', action='store_true')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = False
t0 = time.time()
PROJ = '/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction'
df = pd.read_parquet(PROJ + '/data/de_train.parquet'); idm = pd.read_csv(PROJ + '/data/id_map.csv')
META = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
genes = [c for c in df.columns if c not in META]; G = len(genes)
cells = sorted(df.cell_type.unique()); comps = sorted(set(df.sm_name) | set(idm.sm_name))
CI = {c: i for i, c in enumerate(cells)}; DI = {d: i for i, d in enumerate(comps)}
c_row = df.cell_type.map(CI).to_numpy(); d_row = df.sm_name.map(DI).to_numpy(); ctrl = df.control.to_numpy(bool)
Ynp = df[genes].to_numpy(np.float64); N = len(df)
TNK = [CI[c] for c in ('NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells')]
SPARSE = [CI['B cells'], CI['Myeloid cells']]
row_of = {(c, d): i for i, (c, d) in enumerate(zip(c_row, d_row))}
ANC = sorted(d for d in set(d_row[(c_row == SPARSE[0]) & ~ctrl]) if (SPARSE[1], d) in row_of)
CTRLC = sorted(set(d_row[ctrl])); REF = set(ANC) | set(CTRLC)
QOFF = set(DI[s] for s in idm.sm_name)
Y = torch.tensor(Ynp, dtype=torch.float32, device=dev)
nocd8 = [CI['NK cells'], CI['T cells CD4+'], CI['T regulatory cells']]


def rms_np(x, axis=-1): return np.sqrt(np.mean(np.square(x), axis=axis))


cut = np.quantile([rms_np(Ynp[[row_of[(c, d)] for c in nocd8]].mean(0)) for d in QOFF], .9)


def loo_te(Z, vis, key):
    te = np.zeros_like(Z)
    for k in np.unique(key):
        rows = np.flatnonzero(key == k); v = rows[vis[rows]]
        S = Z[v].sum(0); n = len(v)
        for i in rows:
            if vis[i]: te[i] = (S - Z[i]) / (n - 1) if n > 1 else 0.
            else: te[i] = S / n if n else 0.
    return te


def prepare(hidden):
    vis = np.ones(N, bool); vis[hidden] = False
    Xv = Y[torch.tensor(np.flatnonzero(vis), device=dev)]
    _, _, Vh = torch.linalg.svd(Xv, full_matrices=False); V = Vh[:args.K].T.contiguous()
    Zt = Y @ V; res = (Y.square().sum(1) - Zt.square().sum(1)).clamp(min=0)
    Z = Zt.double().cpu().numpy()
    cte, dte = loo_te(Z, vis, c_row), loo_te(Z, vis, d_row)
    # compound profiles excluding a given cell type: mean over OTHER visible T/NK rows of the compound
    prof = {}
    for c in range(len(cells)):
        Pc = np.zeros((len(comps), args.K)); cnt = np.zeros(len(comps))
        for s in TNK:
            if s == c: continue
            for d in range(len(comps)):
                r = row_of.get((s, d))
                if r is not None and vis[r]: Pc[d] += Z[r]; cnt[d] += 1
        prof[c] = Pc / np.maximum(cnt, 1)[:, None]
    ate = np.zeros_like(Z)
    for i in range(N):
        c, d = c_row[i], d_row[i]
        ref = [row_of[(c, a)] for a in REF if a != d and (c, a) in row_of and vis[row_of[(c, a)]]]
        Pr = prof[c][[d_row[r] for r in ref]]; pm = Pr.mean(0); Pc_ = Pr - pm; q = prof[c][d] - pm
        Km = Pc_ @ Pc_.T; w = np.linalg.solve(Km + args.alpha * np.trace(Km) / len(ref) * np.eye(len(ref)), Pc_ @ q)
        Zr = Z[ref]; zm = Zr.mean(0); ate[i] = zm + w @ (Zr - zm)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    return dict(vis=vis, V=V, z=Zt, res=res, cte=T(cte), dte=T(dte), ate=T(ate))


class Net(torch.nn.Module):
    def __init__(self, E, nin, K, H, mag, zlog_med, dlog_med):
        super().__init__()
        def P(*s, scale): return torch.nn.Parameter((torch.rand(*s, device=dev) * 2 - 1) * scale)
        self.mag = mag
        self.W1 = P(E, nin, H, scale=nin ** -0.5); self.b1 = P(E, 1, H, scale=0.)
        self.W2 = P(E, H, H, scale=H ** -0.5); self.b2 = P(E, 1, H, scale=0.)
        if mag:
            self.Wu = P(E, H, K, scale=H ** -0.5); self.bu = P(E, 1, K, scale=0.)
            self.Wm = P(E, H, 1, scale=0.01 * H ** -0.5)
            self.bm = torch.nn.Parameter(torch.full((E, 1, 1), float(zlog_med - dlog_med), device=dev))
            self.beta = torch.nn.Parameter(torch.ones(E, len(cells), device=dev))
            self.gam = torch.nn.Parameter(torch.zeros(E, len(cells), device=dev))
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


def run(S, hidden, qry, use_ate, mag, seed):
    vis = S['vis']; tr = torch.tensor(np.flatnonzero(vis), device=dev); q = torch.tensor(qry, device=dev)
    feats = [S['cte'], S['dte']] + ([S['ate']] if use_ate else [])
    X = torch.cat(feats, 1); fs = X[tr].std(0) + 1e-6
    ci_all = torch.tensor(c_row, device=dev)
    zt, rt = S['z'][tr], S['res'][tr]; s = zt.std(0) + 1e-3
    zlog = torch.log(zt.norm(dim=1) + 1e-6); zmax = float(zlog.max()) + 1.0
    dnorm = torch.log(S['dte'].norm(dim=1) + 1e-3)
    torch.manual_seed(seed)
    m = Net(args.E, X.shape[1], args.K, args.hidden, mag, float(zlog.median()), float(dnorm[tr].median()))
    dec = [p_ for n, p_ in m.named_parameters() if n.startswith('W')]
    rest = [p_ for n, p_ in m.named_parameters() if not n.startswith('W')]
    opt = torch.optim.AdamW([{'params': dec, 'weight_decay': args.wd}, {'params': rest, 'weight_decay': 0.}], lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xtr, citr = X[tr], ci_all[tr]
    m.train()
    for _ in range(args.epochs):
        noise = 1 + args.sigma * torch.randn(args.E, *Xtr.shape, device=dev)      # LOO-encoder style multiplicative noise
        Xn = Xtr.unsqueeze(0) * noise
        dl = torch.log((S['dte'][tr].unsqueeze(0) * noise[..., args.K:2 * args.K]).norm(dim=-1) + 1e-3)
        out = m(Xn / fs, citr, dl, s, args.dropout, zmax)
        loss = torch.sqrt(((out - zt).square().sum(-1) + rt) / G + 1e-8).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        Xq = (X[q] / fs).unsqueeze(0).expand(args.E, -1, -1)
        zq = m(Xq, ci_all[q], dnorm[q].unsqueeze(0).expand(args.E, -1), s, 0., zmax).median(0).values
    return zq


VARIANTS = {'base': (False, False), 'base_I1': (True, False), 'base_I2': (False, True), 'base_I1_I2': (True, True)}

if args.final:
    kag = idm.sort_values(['cell_type', 'sm_name']).reset_index(drop=True)          # Kaggle id order
    ec, ed = kag.cell_type.map(CI).to_numpy(), kag.sm_name.map(DI).to_numpy()
    n0 = N
    c_row = np.concatenate([c_row, ec]); d_row = np.concatenate([d_row, ed]); ctrl = np.concatenate([ctrl, np.zeros(len(kag), bool)])
    for k_, (c_, d_) in enumerate(zip(ec, ed)):
        assert (c_, d_) not in row_of; row_of[(c_, d_)] = n0 + k_
    qry = np.arange(n0, n0 + len(kag)); use_ate, mag = VARIANTS[args.final]

    def export(fill, seeds):
        global Ynp, Y, N
        Ynp = np.vstack([Ynp[:n0], np.full((len(kag), G), fill)]); N = len(Ynp)
        Y = torch.tensor(Ynp, dtype=torch.float32, device=dev)
        S = prepare(qry)
        zs = [run(S, qry, qry, use_ate, mag, args.seed + 1000 * r) for r in range(seeds)]
        return torch.stack(zs), S['V']
    if args.poison_check:                                   # unknown test labels must never influence predictions
        za, _ = export(0.0, 1); zb, _ = export(1e5, 1)
        print('poison check max |dz| =', float((za - zb).abs().max()), '(|z| scale', float(za.abs().max()), ')'); sys.exit(0)
    zs, V = export(0.0, args.seeds)
    zq = zs.median(0).values; pred = (zq @ V.T).double().cpu().numpy()
    per_seed = (zs @ V.T).double().cpu().numpy()
    spread = float(np.mean([rms_np(per_seed[r] - pred).mean() for r in range(len(per_seed))]))
    fr = pd.DataFrame(pred, columns=genes); fr.insert(0, 'id', np.arange(len(kag)))
    path = os.path.join(args.out, f'{args.final}_kaggle_ids.csv'); fr.to_csv(path, index=False, float_format='%.9g')
    back = pd.read_csv(path)
    assert list(back.columns) == ['id'] + genes and list(back.id) == list(range(len(kag)))
    np.testing.assert_allclose(back[genes].to_numpy(), pred, rtol=1e-6, atol=1e-6); assert np.isfinite(pred).all()
    k1 = pd.read_csv('/tmp/sqr_claude_diag/krr_v1_kaggle_ids.csv')[genes].to_numpy() if os.path.exists('/tmp/sqr_claude_diag/krr_v1_kaggle_ids.csv') else None
    for c_ in ('B cells', 'Myeloid cells'):
        m_ = (kag.cell_type == c_).to_numpy()
        msg = f'{c_}: mean pred row RMS {rms_np(pred[m_]).mean():.3f}'
        if k1 is not None: msg += f' | KRR v1 {rms_np(k1[m_]).mean():.3f} | RMS diff vs KRR v1 {rms_np(pred[m_] - k1[m_]).mean():.3f}'
        print(msg)
    print('mean RMS distance of single-seed predictions to the final median:', round(spread, 4))
    import hashlib; print('wrote', path, 'sha256', hashlib.sha256(open(path, 'rb').read()).hexdigest()); sys.exit(0)

folds = []
for o in TNK:
    hid = np.flatnonzero((c_row == o) & ~ctrl & ~np.isin(d_row, ANC))
    folds.append(('A', cells[o], hid, np.array([i for i in hid if d_row[i] in QOFF])))
for t in SPARSE:
    for d in ANC:
        i = row_of[(t, d)]
        w = rms_np(Ynp[[row_of[(c, d)] for c in nocd8]].mean(0)) < cut
        folds.append(('V1', cells[t] + '|' + comps[d] + ('|weak' if w else ''), np.array([i]), np.array([i])))
if args.smoke: folds = folds[:1] + folds[4:6]
recs = []
for k, (proto, name, hid, qry) in enumerate(folds):
    S = prepare(hid); truth = Ynp[qry]
    for v in args.variants.split(','):
        use_ate, mag = VARIANTS[v]
        zq = torch.stack([run(S, hid, qry, use_ate, mag, args.seed + 1000 * r) for r in range(args.seeds)]).median(0).values
        pred = (zq @ S['V'].T).double().cpu().numpy()
        outs = {v: pred}
        if v == 'base':
            outs['base+magic'] = pred * np.minimum(np.abs(pred).mean(1, keepdims=True) ** 0.6, 1.0)
        for vv, pp in outs.items():
            for i, e in zip(qry, rms_np(pp - truth)):
                recs.append(dict(variant=vv, proto=proto, name=name, row=int(i), err=float(e)))
    print(f'fold {k + 1}/{len(folds)} {proto} {name[:38]} [t={time.time() - t0:.0f}s]', flush=True)
R = pd.DataFrame(recs); R.to_csv(os.path.join(args.out, 'rows.csv'), index=False)


def summ(g):
    a = g[g.proto == 'A'].groupby('name').err.mean()
    b = g[g.proto == 'V1'].copy(); b['target'] = b['name'].str.split('|').str[0]; b['weak'] = b['name'].str.endswith('|weak')
    out = dict(A=a.mean(), V1=b.groupby('target').err.mean().mean(), V1w=b[b.weak].groupby('target').err.mean().mean())
    out['A15'] = a.drop('T cells CD8+').mean() if 'T cells CD8+' in a.index else np.nan
    return pd.Series(out)
Sm = R.groupby('variant').apply(summ)
Sm['objective_vs_KRRv1'] = (Sm.V1w / 1.1625 + Sm.A15 / 0.9894) / 2
pd.set_option('display.width', 250)
print('\nreference: KRR v1 V1w 1.1625 V1 1.8898 A15 0.9894 | KRR v2 obj 0.9897 | 0.6 KRRv2 + 0.4 GT-MLP obj 0.9778')
print(Sm.round(4).to_string())
rng = np.random.default_rng(1)
for v in [x for x in Sm.index if x != 'base']:
    for lab, sel in (('V1w', lambda g: g[(g.proto == 'V1') & g['name'].str.endswith('|weak')]), ('V1 ', lambda g: g[g.proto == 'V1']),
                     ('A15', lambda g: g[(g.proto == 'A') & (g['name'] != 'T cells CD8+')])):
        x = sel(R[R.variant == v]).set_index(['name', 'row']).err; y = sel(R[R.variant == 'base']).set_index(['name', 'row']).err
        if len(x) == 0: continue
        d = (x - y.loc[x.index]).to_numpy(); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f'{v:11s} {lab} delta vs base = {d.mean():+.4f}  95% CI [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  n={len(d)}')
print(json.dumps(vars(args))); print(f'DONE [t={time.time() - t0:.0f}s]')
