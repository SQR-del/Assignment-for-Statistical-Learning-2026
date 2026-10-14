#!/usr/bin/env python3
"""Internal validation (no test labels): ablation of the base network, the two innovations, KRR and the blend.

Protocols
  A   : for each T/NK cell type o: hide all of o's rows except the 15 anchors + 2 controls (as for B/Myeloid),
        predict o's official test compounds from everything else.  A15 = A without CD8+ (CD8+ lacks 2 anchors).
  V1  : B / Myeloid leave-one-anchor-out (30 folds); V1w = the anchors that are weak in the T/NK sources
        (source-mean row RMS below the 90th percentile of the official test compounds), like most test queries.
Pre-declared selection objective = mean(V1w / 1.1625, A15 / 0.9894), i.e. relative to KRR v1 (= 1.000).

    python validate.py                     # full run (~6 min on one A40)
    python validate.py --seeds 1 --smoke   # quick check
"""
import os, time, json, argparse
import numpy as np, pandas as pd
from common import PROJ, Table, rms
from krr import krr_predict
from nn import Config, prepare, ensemble_predict

p = argparse.ArgumentParser()
p.add_argument('--out', default=os.path.join(PROJ, 'results', 'validation'))
p.add_argument('--variants', default='base,base_I1,base_I2,base_I1_I2')
p.add_argument('--seeds', type=int, default=3); p.add_argument('--seed', type=int, default=20260930)
p.add_argument('--weights', default='0.4,0.4,0.2', help='blend weights for base_I2, base_I1_I2, KRR v2')
p.add_argument('--smoke', action='store_true')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
cfg = Config(); t0 = time.time()
tab = Table(); Y = tab.Y; row_of = tab.row_of; nC = len(tab.comps)
variants = args.variants.split(','); w = [float(x) for x in args.weights.split(',')]
REF_KRR1 = dict(V1w=1.1625, A15=0.9894)          # KRR v1, fixed before any network was run

cut = np.quantile([rms(Y[[row_of[(c, d)] for c in tab.SRC]].mean(0)) for d in tab.QOFF], .9)
folds = []
for o in tab.TNK:
    hid = np.flatnonzero((tab.c_row == o) & ~tab.ctrl & ~np.isin(tab.d_row, tab.ANC))
    folds.append(('A', tab.cells[o], hid, np.array([i for i in hid if tab.d_row[i] in tab.QOFF])))
for t in tab.SPARSE:
    for d in tab.ANC:
        i = row_of[(t, d)]; weak = rms(Y[[row_of[(c, d)] for c in tab.SRC]].mean(0)) < cut
        folds.append(('V1', tab.cells[t] + '|' + tab.comps[d] + ('|weak' if weak else ''), np.array([i]), np.array([i])))
if args.smoke: folds = folds[:1] + folds[4:6]

recs = []
for k, (proto, name, hid, qry) in enumerate(folds):
    S = prepare(tab, hid, cfg); truth = Y[qry]; outs = {}
    for v in variants:
        outs[v], _ = ensemble_predict(S, tab, qry, v, args.seeds, args.seed, cfg)
    if 'base' in outs:        # 4th-place post-processing (shrink part), kept as a reference
        outs['base+magic'] = outs['base'] * np.minimum(np.abs(outs['base']).mean(1, keepdims=True) ** 0.6, 1.0)
    qc = [tab.d_row[i] for i in qry]
    if proto == 'A':
        o = tab.CI[name]; src = [s for s in tab.SRC if s != o]; anc = tab.ANC
    else:
        o = tab.CI[name.split('|')[0]]; src = tab.SRC; anc = [a for a in tab.ANC if a != tab.DI[name.split('|')[1]]]
    for ver in (1, 2):
        outs[f'krr_v{ver}'] = krr_predict(Y, row_of, nC, o, src, anc, qc, ver, skip=set(tab.CTRLC))
    if 'base_I2' in outs and 'base_I1_I2' in outs:
        outs['blend'] = w[0] * outs['base_I2'] + w[1] * outs['base_I1_I2'] + w[2] * outs['krr_v2']
    for vv, pp in outs.items():
        for i, e in zip(qry, rms(pp - truth)):
            recs.append(dict(variant=vv, proto=proto, name=name, row=int(i), err=float(e)))
    print(f'fold {k + 1}/{len(folds)} {proto} {name[:40]} [t={time.time() - t0:.0f}s]', flush=True)
R = pd.DataFrame(recs); R.to_csv(os.path.join(args.out, 'rows.csv'), index=False)


def summ(g):
    a = g[g.proto == 'A'].groupby('name').err.mean()
    b = g[g.proto == 'V1'].copy(); b['target'] = b['name'].str.split('|').str[0]; b['weak'] = b['name'].str.endswith('|weak')
    out = dict(A=a.mean(), A15=a.drop('T cells CD8+').mean() if 'T cells CD8+' in a.index else a.mean(),
               V1=b.groupby('target').err.mean().mean(), V1w=b[b.weak].groupby('target').err.mean().mean())
    return pd.Series(out)


Sm = pd.DataFrame({v: summ(g) for v, g in R.groupby('variant')}).T
Sm['objective'] = (Sm.V1w / REF_KRR1['V1w'] + Sm.A15 / REF_KRR1['A15']) / 2
Sm = Sm.sort_values('objective')
pd.set_option('display.width', 200)
print('\nlower is better; objective = mean(V1w / 1.1625, A15 / 0.9894)  (KRR v1 = 1.000 by construction)')
print(Sm.round(4).to_string())
rng = np.random.default_rng(1); ci = {}
if 'base' in Sm.index:
    for v in [x for x in Sm.index if x != 'base']:
        for lab, sel in (('V1w', lambda g: g[(g.proto == 'V1') & g['name'].str.endswith('|weak')]),
                         ('A15', lambda g: g[(g.proto == 'A') & (g['name'] != 'T cells CD8+')])):
            x = sel(R[R.variant == v]).set_index(['name', 'row']).err; y = sel(R[R.variant == 'base']).set_index(['name', 'row']).err
            if len(x) == 0: continue
            d = (x - y.loc[x.index]).to_numpy(); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            ci[f'{v} {lab}'] = [float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
            print(f'{v:11s} {lab} delta vs base = {d.mean():+.4f}  95% CI [{ci[f"{v} {lab}"][1]:+.4f}, {ci[f"{v} {lab}"][2]:+.4f}]  n={len(d)}')
json.dump(dict(args=vars(args), config=vars(cfg), summary=Sm.round(6).to_dict('index'), bootstrap_vs_base=ci),
          open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
print(f'DONE [t={time.time() - t0:.0f}s] -> {args.out}')
