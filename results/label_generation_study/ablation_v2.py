#!/usr/bin/env python3
"""ablation.py + label-generation-aware validation and calibration.

Label generation (verified on the raw counts): B/Myeloid TRAINING labels come from a per-cell-type limma fit on the
training wells only (17 compounds + DMSO, df_total ~130); test labels come from a fit with all wells (df_total ~380).
--shift sim   : protocol A with the held-out T/NK cell type's visible rows (15 anchors + 2 controls) replaced by what a
                restricted fit would give: official + (restricted fit - full fit), both fits from our replication.
--shift dfrev : same, but only the deterministic df part: official -> t (df_full) -> slp (df_restricted).
--calib pre   : df re-calibration of the target's restricted-fit rows before training:
                slp -> t with df_restricted -> slp with df_full (df_full estimated from the OTHER T/NK cell types).
--calib post  : the same map applied to the target's predictions instead.
KRR v1 / KRR v2 are evaluated on the same (shifted) folds; blend = 0.4 base_I2 + 0.4 base_I1_I2 + 0.2 KRR v2.
"""
import os, sys, time, json, argparse, pickle
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument('--out', default='/tmp/sqr_claude_diag/abl2')
p.add_argument('--K', type=int, default=64); p.add_argument('--E', type=int, default=8); p.add_argument('--seeds', type=int, default=3)
p.add_argument('--epochs', type=int, default=400); p.add_argument('--hidden', type=int, default=512)
p.add_argument('--dropout', type=float, default=0.3); p.add_argument('--wd', type=float, default=1e-4)
p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--sigma', type=float, default=0.1)
p.add_argument('--alpha', type=float, default=1.0); p.add_argument('--seed', type=int, default=20260930)
p.add_argument('--variants', default='base,base_I2,base_I1_I2')
p.add_argument('--protos', default='A')
p.add_argument('--shift', default='none', choices=['none', 'sim', 'dfrev'])
p.add_argument('--calib', default='none', choices=['none', 'pre', 'post'])
p.add_argument('--final', default='', help='variant to train on ALL rows and export in Kaggle id order')
p.add_argument('--poison_check', action='store_true')
p.add_argument('--save_preds', action='store_true')
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
Y0np = df[genes].to_numpy(np.float64); Ynp = Y0np.copy(); N = len(df)
TNK = [CI[c] for c in ('NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells')]
SPARSE = [CI['B cells'], CI['Myeloid cells']]
row_of = {(c, d): i for i, (c, d) in enumerate(zip(c_row, d_row))}
ANC = sorted(d for d in set(d_row[(c_row == SPARSE[0]) & ~ctrl]) if (SPARSE[1], d) in row_of)
CTRLC = sorted(set(d_row[ctrl])); REF = set(ANC) | set(CTRLC)
QOFF = set(DI[s] for s in idm.sm_name)
Y = torch.tensor(Ynp, dtype=torch.float32, device=dev)
nocd8 = [CI['NK cells'], CI['T cells CD4+'], CI['T regulatory cells']]
SH = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_data.pkl"), "rb"))


def rms_np(x, axis=-1): return np.sqrt(np.mean(np.square(x), axis=axis))


# ---------------- limma p-value <-> moderated t (two-sided), numerically safe for any |slp|
def slp_from_t(t, df_):
    return np.sign(t) * -(stats.t.logsf(np.abs(t), df_) + np.log(2)) / np.log(10)


def t_from_slp(y, df_):
    y = np.asarray(y, np.float64); tgt = -np.abs(y) * np.log(10) - np.log(2)
    lo, hi = np.full(y.shape, -18.0), np.full(y.shape, np.log(1e5))      # bisection on log t
    for _ in range(64):
        mid = (lo + hi) / 2; above = stats.t.logsf(np.exp(mid), df_) > tgt   # sf too large -> t must grow
        lo = np.where(above, mid, lo); hi = np.where(above, hi, mid)
    return np.where(y == 0, 0.0, np.sign(y) * np.exp((lo + hi) / 2))


def calibrate(y, df_from, df_to): return slp_from_t(t_from_slp(y, df_from), df_to)


def dff_est(o):   # df_full of a held-out cell type estimated from the OTHER T/NK full fits
    return float(np.mean([v for k, v in SH['dff'].items() if k != cells[o]]))


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
        if len(ref) < 2: ate[i] = cte[i]; continue
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
    vis = S['vis']; trn = np.flatnonzero(vis)
    tr = torch.tensor(trn, device=dev); q = torch.tensor(qry, device=dev)
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
        noise = 1 + args.sigma * torch.randn(args.E, *Xtr.shape, device=dev)
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


def krr(Yv, t, src, anchors, qcomps, version):
    """KRR v1 (c=.2, alpha=10, raw) / v2 (c=.4, K=64 projection, alpha=3, clip), exactly as the submitted models."""
    c, alpha = (0.2, 10.0) if version == 1 else (0.4, 3.0)
    anchors = [d for d in anchors if all((s, d) in row_of for s in src) and (t, d) in row_of]
    common = [d for d in range(len(comps)) if all((s, d) in row_of for s in src)]
    rS = np.stack([np.median(Yv[[row_of[(s, d)] for d in common]], 0) for s in src])
    YA = Yv[[row_of[(t, d)] for d in anchors]]; rt = np.median(YA, 0)
    XA = np.stack([Yv[[row_of[(s, d)] for d in anchors]] - rS[j] for j, s in enumerate(src)])
    XQ = np.stack([Yv[[row_of[(s, d)] for d in qcomps]] - rS[j] for j, s in enumerate(src)])
    n = len(anchors); FA = XA.transpose(1, 0, 2).reshape(n, -1); FQ = XQ.transpose(1, 0, 2).reshape(len(qcomps), -1); K = FA @ FA.T
    if version == 1:
        xA, xQ = XA.mean(0), XQ.mean(0)
    else:
        Cm = np.concatenate([Yv[[row_of[(s, d)] for d in common]] - rS[j] for j, s in enumerate(src)])
        ev, U = np.linalg.eigh(Cm @ Cm.T); U = U[:, ::-1][:, :64]; ev = ev[::-1][:64]; V = (Cm.T @ U) / np.sqrt(ev)
        xA, xQ = (XA.mean(0) @ V) @ V.T, (XQ.mean(0) @ V) @ V.T
    beta = np.linalg.solve(K + alpha * np.trace(K) / n * np.eye(n), (YA - rt) - c * xA)
    out = rt + c * xQ + (FQ @ FA.T) @ beta
    if version == 2:
        vis = np.concatenate([Yv[[row_of[(s, d)] for d in common]] for s in src] + [YA]); out = np.clip(out, vis.min(0), vis.max(0))
    return out


VARIANTS = {'base': (False, False), 'base_I1': (True, False), 'base_I2': (False, True), 'base_I1_I2': (True, True)}

if args.final:
    kag = idm.sort_values(['cell_type', 'sm_name']).reset_index(drop=True)          # Kaggle id order
    ec, ed = kag.cell_type.map(CI).to_numpy(), kag.sm_name.map(DI).to_numpy()
    n0 = N
    if args.calib == 'pre':                          # B/Myeloid training rows: restricted-fit df -> full-fit df
        dff_all = float(np.mean(list(SH['dff'].values())))
        for i in np.flatnonzero(np.isin(c_row, SPARSE)):
            Ynp[i] = calibrate(Y0np[i], SH['dfs'][cells[c_row[i]]], dff_all)
        print('calibrated B/Myeloid rows with df_full =', round(dff_all, 1), {cells[t]: round(SH['dfs'][cells[t]], 1) for t in SPARSE})
    c_row = np.concatenate([c_row, ec]); d_row = np.concatenate([d_row, ed]); ctrl = np.concatenate([ctrl, np.zeros(len(kag), bool)])
    for k_, (c_, d_) in enumerate(zip(ec, ed)):
        assert (c_, d_) not in row_of; row_of[(c_, d_)] = n0 + k_
    qry = np.arange(n0, n0 + len(kag))
    Ytrain = Ynp[:n0].copy()

    def export(fill, seeds, variant):
        global Ynp, Y, N
        Ynp = np.vstack([Ytrain, np.full((len(kag), G), fill)]); N = len(Ynp)
        Y = torch.tensor(Ynp, dtype=torch.float32, device=dev)
        S = prepare(qry); use_ate, mag = VARIANTS[variant]
        zs = [run(S, qry, qry, use_ate, mag, args.seed + 1000 * r) for r in range(seeds)]
        return torch.stack(zs), S['V']
    variants = args.final.split(',')
    if args.poison_check:
        za, _ = export(0.0, 1, variants[0]); zb, _ = export(1e5, 1, variants[0])
        print('poison check max |dz| =', float((za - zb).abs().max()), '(|z| scale', float(za.abs().max()), ')'); sys.exit(0)
    import hashlib
    preds = {}
    for v in variants:
        zs, V = export(0.0, args.seeds, v); preds[v] = (zs.median(0).values @ V.T).double().cpu().numpy()
    kr = np.full((len(kag), G), np.nan)                               # KRR v2 with all observed target rows as anchors
    Yk = np.vstack([Ytrain, np.full((len(kag), G), np.nan)])
    for t in SPARSE:
        m_ = np.flatnonzero(ec == t); anc = sorted(d_row[:n0][c_row[:n0] == t])
        kr[m_] = krr(Yk, t, nocd8, anc, list(ed[m_]), 2)
    preds['krr_v2'] = kr
    if all(v in preds for v in ('base_I2', 'base_I1_I2')):
        preds['blend'] = 0.4 * preds['base_I2'] + 0.4 * preds['base_I1_I2'] + 0.2 * kr
    ref = {'base_I2': PROJ + '/results/nn_base_I2_v1/base_I2_kaggle_ids.csv', 'base_I1_I2': PROJ + '/results/nn_base_I1_I2_v1/base_I1_I2_kaggle_ids.csv',
           'krr_v2': PROJ + '/results/krr_v2/krr_v2_kaggle_ids.csv', 'blend': PROJ + '/results/blend_v1/blend_v1_kaggle_ids.csv'}
    for name, pred in preds.items():
        assert np.isfinite(pred).all()
        fr = pd.DataFrame(pred, columns=genes); fr.insert(0, 'id', np.arange(len(kag)))
        path = os.path.join(args.out, f'{name}_calib_{args.calib}_kaggle_ids.csv'); fr.to_csv(path, index=False, float_format='%.9g')
        back = pd.read_csv(path); assert list(back.columns) == ['id'] + genes and list(back.id) == list(range(len(kag)))
        np.testing.assert_allclose(back[genes].to_numpy(), pred, rtol=1e-6, atol=1e-6)
        msg = f'{name:11s}'
        for c_ in ('B cells', 'Myeloid cells'):
            m_ = (kag.cell_type == c_).to_numpy(); msg += f' | {c_[:3]} row RMS {rms_np(pred[m_]).mean():.3f}'
        if name in ref and os.path.exists(ref[name]):
            old = pd.read_csv(ref[name])[genes].to_numpy(); msg += f' | old submitted {rms_np(old).mean():.3f}, RMS diff {rms_np(pred - old).mean():.3f}'
        print(msg, '| sha256', hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16], flush=True)
    sys.exit(0)

folds = []
if 'A' in args.protos:
    for o in TNK:
        hid = np.flatnonzero((c_row == o) & ~ctrl & ~np.isin(d_row, ANC))
        folds.append(('A', cells[o], hid, np.array([i for i in hid if d_row[i] in QOFF])))
if 'V1' in args.protos:
    for t in SPARSE:
        for d in ANC:
            i = row_of[(t, d)]
            w = rms_np(Y0np[[row_of[(c, d)] for c in nocd8]].mean(0)) < cut
            folds.append(('V1', cells[t] + '|' + comps[d] + ('|weak' if w else ''), np.array([i]), np.array([i])))


def fold_labels(proto, name, hid):
    """Training labels as the model would see them in this fold (truth always stays the official Y0np)."""
    Yf = Y0np.copy(); info = []
    if proto != 'A': return Yf, info
    o = CI[name]; hs = set(hid.tolist())
    for i in np.flatnonzero(c_row == o):
        if i in hs: continue
        k = (name, comps[d_row[i]])
        if k not in SH['small']: continue
        if args.shift == 'sim' and SH['r_full_off'][k] >= 0.8: Yf[i] = Y0np[i] + SH['small'][k] - SH['full'][k]
        elif args.shift != 'none': Yf[i] = calibrate(Y0np[i], SH['dff'][name], SH['dfs'][name])
        if args.calib == 'pre': Yf[i] = calibrate(Yf[i], SH['dfs'][name], dff_est(o))
        info.append((k[1], rms_np(Y0np[i]), rms_np(Yf[i])))
    return Yf, info


recs = []; PRED = {}
for k, (proto, name, hid, qry) in enumerate(folds):
    Ynp, info = fold_labels(proto, name, hid); Y = torch.tensor(Ynp, dtype=torch.float32, device=dev)
    if info:
        strong = [x for x in info if x[1] > 1.5]
        print(f'  {name}: visible target rows {len(info)}; strong rows official RMS {np.mean([x[1] for x in strong]):.2f} -> seen {np.mean([x[2] for x in strong]):.2f}')
    S = prepare(hid); truth = Y0np[qry]
    outs = {}
    for v in args.variants.split(','):
        use_ate, mag = VARIANTS[v]
        zq = torch.stack([run(S, hid, qry, use_ate, mag, args.seed + 1000 * r) for r in range(args.seeds)]).median(0).values
        outs[v] = (zq @ S['V'].T).double().cpu().numpy()
    if proto == 'A':
        o = CI[name]; src = [s for s in nocd8 if s != o]; qc = [d_row[i] for i in qry]
        outs['krr_v1'] = krr(Ynp, o, src, ANC, qc, 1); outs['krr_v2'] = krr(Ynp, o, src, ANC, qc, 2)
    else:
        t = CI[name.split('|')[0]]; d = DI[name.split('|')[1]]; anc = [a for a in ANC if a != d]
        outs['krr_v1'] = krr(Ynp, t, nocd8, anc, [d], 1); outs['krr_v2'] = krr(Ynp, t, nocd8, anc, [d], 2)
    if all(v in outs for v in ('base_I2', 'base_I1_I2')):
        outs['blend'] = 0.4 * outs['base_I2'] + 0.4 * outs['base_I1_I2'] + 0.2 * outs['krr_v2']
    if args.calib == 'post' and proto == 'A':
        o = CI[name]; outs = {v: calibrate(pp, SH['dfs'][name], dff_est(o)) for v, pp in outs.items()}
    for vv, pp in outs.items():
        if args.save_preds: PRED[(vv, proto, name)] = (qry, pp.astype(np.float32))
        for i, e in zip(qry, rms_np(pp - truth)):
            recs.append(dict(variant=vv, proto=proto, name=name, row=int(i), err=float(e), truth_rms=float(rms_np(Y0np[i])), pred_rms=float(rms_np(pp[list(qry).index(i)]))))
    print(f'fold {k + 1}/{len(folds)} {proto} {name[:38]} [t={time.time() - t0:.0f}s]', flush=True)
R = pd.DataFrame(recs); R.to_csv(os.path.join(args.out, 'rows.csv'), index=False)
if args.save_preds: pickle.dump(PRED, open(os.path.join(args.out, 'preds.pkl'), 'wb'))
A = R[R.proto == 'A']
if len(A):
    T = A.groupby(['variant', 'name']).err.mean().unstack()
    T['A'] = T.mean(1); T['A15'] = T.drop(columns=['A', 'T cells CD8+'], errors='ignore').mean(1)
    T['A15_rel_krr_v1'] = T.A15 / T.loc['krr_v1', 'A15']
    T['pred/truth RMS'] = A.groupby('variant').pred_rms.mean() / A.groupby('variant').truth_rms.mean()
    pd.set_option('display.width', 250)
    print(f'\nprotocol A (shift={args.shift}, calib={args.calib}); official private relative to KRR v1: base 0.980, base_I2 0.907, blend 0.906')
    print(T.round(4).to_string())
print(json.dumps(vars(args))); print(f'DONE [t={time.time() - t0:.0f}s]')
