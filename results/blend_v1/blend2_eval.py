#!/usr/bin/env python3
"""Blend base_I2, base_I1_I2 (saved fold predictions) and KRR v2 on identical folds; pre-declared objective."""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_k] = '8'
import pickle, itertools
import numpy as np, pandas as pd
PROJ = '/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction'
df = pd.read_parquet(PROJ + '/data/de_train.parquet'); idm = pd.read_csv(PROJ + '/data/id_map.csv')
META = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
genes = [c for c in df.columns if c not in META]; G = len(genes)
cells = sorted(df.cell_type.unique()); comps = sorted(set(df.sm_name) | set(idm.sm_name))
CI = {c: i for i, c in enumerate(cells)}; DI = {d: i for i, d in enumerate(comps)}
c_row = df.cell_type.map(CI).to_numpy(); d_row = df.sm_name.map(DI).to_numpy(); ctrl = df.control.to_numpy(bool)
Ynp = df[genes].to_numpy(np.float64)
row_of = {(c, d): i for i, (c, d) in enumerate(zip(c_row, d_row))}
SPARSE = [CI['B cells'], CI['Myeloid cells']]
ANC = sorted(d for d in set(d_row[(c_row == SPARSE[0]) & ~ctrl]) if (SPARSE[1], d) in row_of)
nocd8 = [CI['NK cells'], CI['T cells CD4+'], CI['T regulatory cells']]
rms = lambda x: np.sqrt(np.mean(np.square(x), -1))


def krr_v2(t, src, anchors, qcomps, c=0.4, Kd=64, alpha=3.0):
    anchors = [d for d in anchors if all((s, d) in row_of for s in src) and (t, d) in row_of]
    common = [d for d in range(len(comps)) if all((s, d) in row_of for s in src)]
    rS = np.stack([np.median(Ynp[[row_of[(s, d)] for d in common]], 0) for s in src])
    Cm = np.concatenate([Ynp[[row_of[(s, d)] for d in common]] - rS[j] for j, s in enumerate(src)])
    ev, U = np.linalg.eigh(Cm @ Cm.T); U = U[:, ::-1][:, :Kd]; ev = ev[::-1][:Kd]; V = (Cm.T @ U) / np.sqrt(ev)
    YA = Ynp[[row_of[(t, d)] for d in anchors]]; rt = np.median(YA, 0)
    XA = np.stack([Ynp[[row_of[(s, d)] for d in anchors]] - rS[j] for j, s in enumerate(src)])
    XQ = np.stack([Ynp[[row_of[(s, d)] for d in qcomps]] - rS[j] for j, s in enumerate(src)])
    n = len(anchors); FA = XA.transpose(1, 0, 2).reshape(n, -1); FQ = XQ.transpose(1, 0, 2).reshape(len(qcomps), -1); K = FA @ FA.T
    xA, xQ = (XA.mean(0) @ V) @ V.T, (XQ.mean(0) @ V) @ V.T
    beta = np.linalg.solve(K + alpha * np.trace(K) / n * np.eye(n), (YA - rt) - c * xA)
    vis = np.concatenate([Ynp[[row_of[(s, d)] for d in common]] for s in src] + [YA])
    return np.clip(rt + c * xQ + (FQ @ FA.T) @ beta, vis.min(0), vis.max(0))


import sys
PATH = sys.argv[1] if len(sys.argv) > 1 else '/tmp/sqr_claude_diag/imp_R0'
FIXED = sys.argv[2] if len(sys.argv) > 2 else ''
P = pickle.load(open(PATH + '/preds.pkl', 'rb'))
KC = '/tmp/sqr_claude_diag/krr_v2_fold_cache.pkl'
KCACHE = pickle.load(open(KC, 'rb')) if os.path.exists(KC) else {}
keys = sorted({(pr, nm) for (_, pr, nm) in P})
F = {}
for pr, nm in keys:
    qry, p1 = P[('base_I2', pr, nm)]; _, p2 = P[('base_I1_I2', pr, nm)]
    if (pr, nm) in KCACHE: pk = KCACHE[(pr, nm)]
    elif pr == 'A':
        o = CI[nm]; src = [s for s in nocd8 if s != o]; pk = krr_v2(o, src, ANC, [d_row[i] for i in qry])
    else:
        t = CI[nm.split('|')[0]]; d = DI[nm.split('|')[1]]; pk = krr_v2(t, nocd8, [a for a in ANC if a != d], [d])
    KCACHE[(pr, nm)] = pk
    F[(pr, nm)] = (qry, p1.astype(np.float64), p2.astype(np.float64), pk, Ynp[qry])
print('folds', len(F)); pickle.dump(KCACHE, open(KC, 'wb'))


def score(w):
    A, B = {}, {}
    for (pr, nm), (qry, p1, p2, pk, y) in F.items():
        e = rms(w[0] * p1 + w[1] * p2 + w[2] * pk - y)
        if pr == 'A': A[nm] = e.mean()
        else: B.setdefault(nm.split('|')[0], []).append((e[0], nm.endswith('|weak')))
    A15 = np.mean([v for k, v in A.items() if k != 'T cells CD8+'])
    V1 = np.mean([np.mean([e for e, _ in v]) for v in B.values()]); V1w = np.mean([np.mean([e for e, w_ in v if w_]) for v in B.values()])
    return dict(A=np.mean(list(A.values())), A15=A15, V1=V1, V1w=V1w, objective=(V1w / 1.1625 + A15 / 0.9894) / 2)


if FIXED:
    for w in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), tuple(float(x) for x in FIXED.split(','))]:
        print(PATH.split('/')[-1], 'weights', w, {k: round(v, 4) for k, v in score(w).items()})
    raise SystemExit
grid = [w for w in itertools.product(np.round(np.arange(0, 1.01, .1), 1), repeat=3) if abs(sum(w) - 1) < 1e-9]
T = pd.DataFrame([dict(w_I2=w[0], w_I1I2=w[1], w_KRR=w[2], **score(w)) for w in grid]).sort_values('objective')
pd.set_option('display.width', 200)
print('reference single models:'); print(T[(T.w_I2 == 1) | (T.w_I1I2 == 1) | (T.w_KRR == 1)].round(4).to_string(index=False))
print('\nbest 12 blends by pre-declared objective:'); print(T.head(12).round(4).to_string(index=False))
T.to_csv('/tmp/sqr_claude_diag/imp_R0/blend_grid.csv', index=False)
