#!/usr/bin/env python3
"""Null-centred multi-output kernel ridge over compounds (anchor-analog transfer).

Prediction for target t, query compound d (raw signed -log10 p scale):
    yhat = r_t + c * xbar_d + k(d, A) (K_AA + alpha * tr(K)/n * I)^{-1} [Y_tA - r_t - c * xbar_A]
r_t  : per-gene median over the target's anchors (null reference)
x_sd : source response minus that source's per-gene median over all compounds; features concatenate sources
xbar : mean over sources of x_sd;  K, k : linear kernels on concatenated source features.
Hyperparameters (c, alpha) are chosen only on the 12 protocol-A inner tasks (T/NK data).
Final fit: sources NK / CD4+ / Treg (CD8 excluded so that all 15 target anchors are usable).
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = os.environ.get('KRR_THREADS', '8')
import sys, json, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path('/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction')
sys.path.insert(0, str(PROJ / 'code'))
from experiment import load_data, file_hash  # noqa: E402

OUT = PROJ / 'results' / 'krr_null_v1'
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'submissions').mkdir()
t0 = time.time()
D = load_data(PROJ / 'data/de_train.parquet', PROJ / 'data/id_map.csv')
Y, OBS, E, P = D.y, D.observed, D.environments, D.perturbations
G = Y.shape[2]; ANC, DENSE, SPARSE = D.anchors, D.dense, D.sparse
CD8 = E.index('T cells CD8+'); NOCD8 = [s for s in DENSE if s != CD8]
CS = (0.0, 0.1, 0.2, 0.3, 0.4)
ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
GRID = [(c, a) for c in CS for a in ALPHAS]


def rms(x, axis=-1): return np.sqrt(np.mean(np.square(x), axis=axis))


def matched(t, src, anc): return anc[OBS[t, anc] & OBS[np.ix_(src, anc)].all(0)]


def A_queries(t, src):
    el = OBS[t].copy(); el[ANC] = False; el &= OBS[list(src)].all(0); return np.flatnonzero(el)


class Fit:
    """Sufficient statistics; only target anchors and source rows are read."""

    def __init__(self, Yv, t, src, anchors, queries):
        src = list(src); self.anchors = matched(t, src, np.asarray(anchors)); q = np.asarray(queries)
        if len(self.anchors) < 3: raise ValueError('too few anchors')
        if not OBS[np.ix_(src, q)].all(): raise ValueError('query lacks a source')
        common = np.flatnonzero(OBS[src].all(0))
        rS = np.median(Yv[np.ix_(src, common)], axis=1)
        YA = Yv[t, self.anchors]
        self.rt = np.median(YA, 0)
        XA = Yv[np.ix_(src, self.anchors)] - rS[:, None]; XQ = Yv[np.ix_(src, q)] - rS[:, None]
        n, m = len(self.anchors), len(q)
        FA = XA.transpose(1, 0, 2).reshape(n, -1); FQ = XQ.transpose(1, 0, 2).reshape(m, -1)
        self.K, self.kq = FA @ FA.T, FQ @ FA.T
        self.YAc, self.xA, self.xQ = YA - self.rt, XA.mean(0), XQ.mean(0)
        for a in (YA, XA, XQ):
            if not np.isfinite(a).all(): raise ValueError('non-finite visible input')

    def predict(self, c, alpha):
        n = len(self.K)
        beta = np.linalg.solve(self.K + alpha * np.trace(self.K) / n * np.eye(n), self.YAc - c * self.xA)
        return self.rt + c * self.xQ + self.kq @ beta


def grid_scores(fit, truth): return np.array([rms(fit.predict(c, a) - truth).mean() for c, a in GRID])


# ---------------- hyperparameter selection: protocol A inner tasks only
inner = {}
for o in DENSE:
    pool = [s for s in DENSE if s != o]
    for i in pool:
        src = [s for s in pool if s != i]; q = A_queries(i, src)
        inner[(o, i)] = grid_scores(Fit(Y, i, src, ANC, q), Y[i, q])
glob = GRID[int(np.argmin(np.mean(list(inner.values()), 0)))]
print('global (c, alpha) from 12 A inner tasks:', glob, flush=True)
grid_table = pd.DataFrame([dict(outer=E[o], inner=E[i], c=c, alpha=a, MRRMSE=s)
                           for (o, i), sc in inner.items() for (c, a), s in zip(GRID, sc)])
grid_table.to_csv(OUT / 'A_inner_grid.csv', index=False)

# ---------------- internal evaluation (no B/Myeloid query labels exist; B = anchor leave-one-out)
A_rows = []
for o in DENSE:
    pool = [s for s in DENSE if s != o]
    h = GRID[int(np.argmin(np.mean([inner[(o, i)] for i in pool], 0)))]
    q = A_queries(o, pool); f = Fit(Y, o, pool, ANC, q)
    err = rms(f.predict(*h) - Y[o, q]); A_rows.append(dict(target=E[o], c=h[0], alpha=h[1], MRRMSE=err.mean()))
    # leakage check: hidden outer-target values replaced by huge random numbers must not change predictions
    Yp = Y.copy(); hidden = np.ones(Y.shape[1], bool); hidden[f.anchors] = False
    Yp[o, hidden] = np.random.default_rng(o).normal(1e5, 1e3, size=(hidden.sum(), G))
    np.testing.assert_allclose(Fit(Yp, o, pool, ANC, q).predict(*h), f.predict(*h), rtol=1e-10, atol=1e-10)
A_nested = float(np.mean([r['MRRMSE'] for r in A_rows]))
A15 = []
for o in NOCD8:
    src = [s for s in NOCD8 if s != o]; q = A_queries(o, src)
    A15.append(rms(Fit(Y, o, src, ANC, q).predict(*glob) - Y[o, q]).mean())
B_rows = []
for t in SPARSE:
    anc = np.flatnonzero(OBS[t])
    for q in anc:
        f = Fit(Y, t, NOCD8, anc[anc != q], [q])
        B_rows.append(dict(target=E[t], anchor=P[q], rmse=float(rms(f.predict(*glob) - Y[t, [q]])[0])))
Bdf = pd.DataFrame(B_rows); Bdf.to_csv(OUT / 'B_loo_scores.csv', index=False)
B_macro = float(Bdf.groupby('target').rmse.mean().mean())
print('A nested', round(A_nested, 4), '| A15', round(float(np.mean(A15)), 4), '| B LOO (no CD8)', round(B_macro, 4), flush=True)

# ---------------- final fit and export in id_map order
idm = D.id_map
ei = {x: i for i, x in enumerate(E)}; di = {x: i for i, x in enumerate(P)}
pred = np.full((len(idm), G), np.nan)
final_audit = []
for t in SPARSE:
    rows = np.flatnonzero((idm.cell_type == E[t]).to_numpy())
    q = np.array([di[str(s)] for s in idm.sm_name.iloc[rows]])
    if OBS[t, q].any(): raise ValueError('query overlaps target training rows')
    f = Fit(Y, t, NOCD8, np.flatnonzero(OBS[t]), q)
    pred[rows] = f.predict(*glob)
    # invariance: CD8 rows and all non-anchor target entries are never read
    Yp = Y.copy(); Yp[CD8] = 7.0; Yp[t, ~OBS[t]] = 1e5
    np.testing.assert_allclose(Fit(Yp, t, NOCD8, np.flatnonzero(OBS[t]), q).predict(*glob), pred[rows], rtol=1e-10, atol=1e-10)
    final_audit.append(dict(target=E[t], sources=[E[s] for s in NOCD8], anchors=[P[a] for a in f.anchors],
                            n_rows=len(rows), mean_pred_row_RMS=float(rms(pred[rows]).mean()),
                            null_reference_RMS=float(rms(f.rt))))
if not np.isfinite(pred).all(): raise ValueError('unfilled predictions')
frame = pd.DataFrame(pred, columns=D.responses); frame.insert(0, 'id', idm.id.to_numpy())
path = OUT / 'submissions' / 'krr_null.csv'
frame.to_csv(path, index=False, float_format='%.9g')
back = pd.read_csv(path)
assert list(back.columns) == ['id'] + D.responses and np.array_equal(back.id.to_numpy(), idm.id.to_numpy())
np.testing.assert_allclose(back[D.responses].to_numpy(), pred, rtol=1e-8, atol=1e-8)
ref = pd.read_csv(PROJ / 'results/lgt_20260930_v1/submissions/L3.csv')
assert list(ref.columns) == list(back.columns) and np.array_equal(ref.id.to_numpy(), back.id.to_numpy())
L3 = ref[D.responses].to_numpy()
cmp = {E[t]: dict(KRR_pred_RMS=float(rms(pred[m_]).mean()), L3_pred_RMS=float(rms(L3[m_]).mean()),
                  RMS_diff_KRR_vs_L3=float(rms(pred[m_] - L3[m_]).mean()))
       for t in SPARSE for m_ in [(idm.cell_type == E[t]).to_numpy()]}
shutil.copy(__file__, OUT / 'predict_krr.py')
summary = dict(model='null-centred multi-output KRR over compounds (raw scale)', selected=dict(c=glob[0], alpha=glob[1]),
               grid=dict(c=list(CS), alpha=list(ALPHAS)), selection='mean MRRMSE over the 12 protocol-A inner tasks',
               internal=dict(A_nested=A_nested, A_nested_by_target=A_rows, A15=float(np.mean(A15)), B_LOO_noCD8=B_macro,
                             reference_repo_L3=dict(A_nested=1.0395, A15=1.3358, B_LOO_noCD8=2.1744)),
               final_fit=final_audit, comparison_with_repo_L3=cmp,
               checks=['hidden outer-target poison invariance (4 folds)', 'final fit invariant to CD8 rows and hidden target entries',
                       'CSV readback exact', 'ids and columns identical to repo L3.csv'],
               submission=dict(path=str(path.relative_to(OUT)), rows=len(frame), columns=len(frame.columns),
                               bytes=path.stat().st_size, sha256=file_hash(path)),
               inputs=D.provenance, official_kaggle_score=None, elapsed_seconds=time.time() - t0)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: summary[k] for k in ('selected', 'internal', 'comparison_with_repo_L3', 'submission')}, indent=1, ensure_ascii=False))
print(f'DONE [t={time.time() - t0:.0f}s]')
