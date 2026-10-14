#!/usr/bin/env python3
"""Control experiment: kernel ridge on the network's inputs vs the network, on the same 34 folds (methods/harness.py).
Grid: inputs {base, base+I1} x kernel {linear, rbf(gamma0)} x loss {mse, mrrmse} x lam. Writes grid.csv + summary."""
import os, sys, time, json
PROJ = '/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction'
sys.path.insert(0, PROJ + '/methods'); sys.path.insert(0, PROJ + '/code')
import numpy as np, pandas as pd
from harness import folds, masked, summarize
from common import Table, rms
from nn import Config, prepare
from krr_joint import KRRJoint

OUT = PROJ + '/results/krr_joint_study'; os.makedirs(OUT, exist_ok=True)
LAMS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
KERNELS = [('linear', None)] + [('rbf', g) for g in (0.25, 0.5, 1.0, 2.0)]
tab = Table(); Y = tab.Y; t0 = time.time(); recs = []
for k, (proto, name, hid, qry) in enumerate(folds(tab)):
    S = prepare(masked(tab, hid), hid, Config())
    for use_ate in (False, True):
        for kind, g in KERNELS:
            m = KRRJoint(S, tab.G, kind, g or 1.0, use_ate)
            for loss in ('mse', 'mrrmse'):
                for lam in LAMS:
                    pred = m.predict(qry, lam, loss); assert np.isfinite(pred).all()
                    cfg = f"{'I1' if use_ate else 'base'}|{kind}{'' if g is None else f'(g={g})'}|{loss}|lam={lam:g}"
                    recs += [dict(cfg=cfg, proto=proto, name=name, row=int(i), err=float(e)) for i, e in zip(qry, rms(pred - Y[qry]))]
    print(f'fold {k + 1}/34 {proto} {name[:36]} [t={time.time() - t0:.0f}s]', flush=True)
R = pd.DataFrame(recs); R.to_csv(OUT + '/grid_rows.csv.gz', index=False)
T = pd.DataFrame({c: summarize(g) for c, g in R.groupby('cfg')}).T
T[['inputs', 'kernel', 'loss', 'lam']] = pd.Series(T.index).str.split('|', expand=True).values
T = T.sort_values('objective'); T.to_csv(OUT + '/grid.csv')
pd.set_option('display.width', 220)
cols = ['A15', 'V1w', 'V1', 'objective']
print('\nreference (same folds): NN base obj 0.957 (A15 0.988, V1w 1.064, V1 2.088) | NN base_I1 0.951 | KRR v1 1.000')
print('\nbest config per (inputs, kernel family, loss):')
T['family'] = T.kernel.str.replace(r'\(g=.*\)', '', regex=True)
print(T.groupby(['inputs', 'family', 'loss'], sort=False).head(1)[cols].round(4).to_string())
print('\nmedian over the grid per (inputs, family, loss):')
print(T.groupby(['inputs', 'family', 'loss'])[cols].median().round(4).to_string())
print(f'DONE [t={time.time() - t0:.0f}s]')
