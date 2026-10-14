#!/usr/bin/env python3
"""KRR v2 (same framework): null-centred anchor-analog KRR + denoised same-compound transfer + clipping.
yhat = clip( r_t + c P_K xbar_d + k(d,A)(K + alpha tr(K)/n I)^{-1}[Y_tA - r_t - c P_K xbar_A] )
Config chosen by the pre-declared rule in diag7 (V1w + A15 objective): c=0.4, K=64, alpha=3, clip=True.
Sources NK / CD4+ / Treg; all 15 target anchors. Output rows in KAGGLE id order (sorted by cell_type, sm_name).
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_k] = '8'
import sys, json, shutil, hashlib
from pathlib import Path
import numpy as np, pandas as pd
PROJ = Path('/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction'); sys.path.insert(0, str(PROJ / 'code'))
from experiment import load_data
C, KDIM, ALPHA = 0.4, 64, 3.0
OUT = PROJ / 'results' / 'krr_v2'; OUT.mkdir(parents=True, exist_ok=False)
D = load_data(PROJ / 'data/de_train.parquet', PROJ / 'data/id_map.csv')
Y, OBS, E, P = D.y, D.observed, D.environments, D.perturbations; G = Y.shape[2]
DENSE, SPARSE = D.dense, D.sparse; CD8 = E.index('T cells CD8+'); SRC = [s for s in DENSE if s != CD8]
PI = {p: i for i, p in enumerate(P)}
kag = D.id_map.sort_values(['cell_type', 'sm_name']).reset_index(drop=True)   # Kaggle id order
kag['kaggle_id'] = np.arange(len(kag))


def rms(x, axis=-1): return np.sqrt(np.mean(np.square(x), axis=axis))


common = np.flatnonzero(OBS[SRC].all(0))
rS = np.median(Y[np.ix_(SRC, common)], 1)
Cm = (Y[np.ix_(SRC, common)] - rS[:, None]).reshape(-1, G)
ev, U = np.linalg.eigh(Cm @ Cm.T); U = U[:, ::-1][:, :KDIM]; ev = ev[::-1][:KDIM]
V = (Cm.T @ U) / np.sqrt(ev)


def fit_predict(Yv, t, queries):
    anc = np.flatnonzero(OBS[t]); YA = Yv[t, anc]; rt = np.median(YA, 0)
    XA = Yv[np.ix_(SRC, anc)] - rS[:, None]; XQ = Yv[np.ix_(SRC, queries)] - rS[:, None]; n, m = len(anc), len(queries)
    FA = XA.transpose(1, 0, 2).reshape(n, -1); FQ = XQ.transpose(1, 0, 2).reshape(m, -1); K = FA @ FA.T
    xA, xQ = (XA.mean(0) @ V) @ V.T, (XQ.mean(0) @ V) @ V.T
    beta = np.linalg.solve(K + ALPHA * np.trace(K) / n * np.eye(n), (YA - rt) - C * xA)
    vis = np.concatenate([Yv[s, OBS[s]] for s in SRC] + [YA])
    return np.clip(rt + C * xQ + (FQ @ FA.T) @ beta, vis.min(0), vis.max(0))


pred = np.full((len(kag), G), np.nan)
for t in SPARSE:
    rows = np.flatnonzero((kag.cell_type == E[t]).to_numpy())
    q = np.array([PI[s] for s in kag.sm_name.iloc[rows]])
    assert not OBS[t, q].any() and OBS[np.ix_(SRC, q)].all()
    pred[rows] = fit_predict(Y, t, q)
    Yp = Y.copy(); Yp[CD8] = 7.0; Yp[t, ~OBS[t]] = 1e5          # unread entries must not matter
    np.testing.assert_allclose(fit_predict(Yp, t, q), pred[rows], rtol=1e-10, atol=1e-10)
assert np.isfinite(pred).all()
frame = pd.DataFrame(pred, columns=D.responses); frame.insert(0, 'id', kag.kaggle_id.to_numpy())
path = OUT / 'krr_v2_kaggle_ids.csv'; frame.to_csv(path, index=False, float_format='%.9g')
back = pd.read_csv(path); np.testing.assert_allclose(back[D.responses].to_numpy(), pred, rtol=1e-8, atol=1e-8)
assert list(back.id) == list(range(255))
v1 = pd.read_csv(PROJ / 'results/kaggle_id_fix/krr_null_v1_kaggle_ids.csv')
assert list(v1.columns) == list(back.columns)
old = v1[D.responses].to_numpy()
cmp = {E[t]: dict(pred_row_RMS_v2=float(rms(pred[m_]).mean()), pred_row_RMS_v1=float(rms(old[m_]).mean()),
                  RMS_diff=float(rms(pred[m_] - old[m_]).mean()))
       for t in SPARSE for m_ in [(kag.cell_type == E[t]).to_numpy()]}
shutil.copy(__file__, OUT / 'predict_krr_v2.py')
info = dict(config=dict(c=C, K=KDIM, alpha=ALPHA, clip=True), sources=[E[s] for s in SRC], id_order='kaggle (cell_type, sm_name)',
            internal=dict(V1w=1.1325, V1=1.7311, A15=0.9946, A=0.9189, objective=0.9897,
                          v1_reference=dict(V1w=1.1625, V1=1.8898, A15=0.9894, A=0.9197)),
            comparison_with_v1=cmp, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
(OUT / 'summary.json').write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(info, indent=1, ensure_ascii=False))
