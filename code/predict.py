#!/usr/bin/env python3
"""Final model: train on all 614 training rows and write the 255 test predictions in Kaggle id order.

blend = 0.4 * (base + I2) + 0.4 * (base + I1 + I2) + 0.2 * KRR v2
    networks: 10 seeds x ensembles of 8, median over seeds; KRR v2: sources NK / CD4+ / Treg, the 15 target anchors.
Leakage check (always run): the placeholder labels of the test rows are set to 0 and to 1e5; every component must
give identical predictions, i.e. unknown test labels cannot influence the output.

    python predict.py                        # -> results/blend_final/ (~3 min on one A40)
"""
import os, json, time, hashlib, argparse
import numpy as np, pandas as pd
from common import PROJ, Table, rms
from krr import krr_predict
from nn import Config, prepare, ensemble_predict

p = argparse.ArgumentParser()
p.add_argument('--out', default=os.path.join(PROJ, 'results', 'blend_final'))
p.add_argument('--seeds', type=int, default=10); p.add_argument('--seed', type=int, default=20260930)
p.add_argument('--weights', default='0.4,0.4,0.2', help='weights for base_I2, base_I1_I2, KRR v2')
p.add_argument('--compare', default=os.path.join(PROJ, 'results', 'blend_v1'), help='directory of a previous export to compare with')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
cfg = Config(); t0 = time.time(); w = [float(x) for x in args.weights.split(',')]
tab0 = Table()


def components(fill, seeds):
    tab, qry = tab0.with_queries(fill)
    S = prepare(tab, qry, cfg); out = {}
    for v in ('base_I2', 'base_I1_I2'):
        out[v], _ = ensemble_predict(S, tab, qry, v, seeds, args.seed, cfg)
    kr = np.full((len(qry), tab.G), np.nan); ed = tab.d_row[qry]
    for t in tab.SPARSE:
        m = np.flatnonzero(tab.c_row[qry] == t)
        anc = sorted(tab.d_row[:tab.n_train][tab.c_row[:tab.n_train] == t])       # observed rows of t (controls skipped)
        kr[m] = krr_predict(tab.Y, tab.row_of, len(tab.comps), t, tab.SRC, anc, list(ed[m]), 2, skip=set(tab.CTRLC))
    out['krr_v2'] = kr
    return out


# leakage check with one seed per network
a, b = components(0.0, 1), components(1e5, 1)
leak = {k: float(np.abs(a[k] - b[k]).max()) for k in a}
print('leakage check (max |pred(fill=0) - pred(fill=1e5)|):', leak, flush=True)
assert all(v == 0.0 for v in leak.values()), 'test placeholders influence predictions'

P = components(0.0, args.seeds)
P['blend'] = w[0] * P['base_I2'] + w[1] * P['base_I1_I2'] + w[2] * P['krr_v2']
info = dict(model='0.4 * base_I2 + 0.4 * base_I1_I2 + 0.2 * KRR v2', weights=w, seeds=args.seeds, base_seed=args.seed,
            config=vars(cfg), leakage_check=leak, files={})
for name, pred in P.items():
    path = os.path.join(args.out, f'{name}_kaggle_ids.csv'); tab0.write_submission(pred, path)
    rec = dict(sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest(),
               row_RMS={c: float(rms(pred[(tab0.kaggle.cell_type == c).to_numpy()]).mean()) for c in ('B cells', 'Myeloid cells')})
    ref = {'blend': os.path.join(args.compare, 'blend_v1_kaggle_ids.csv'),
           'base_I2': os.path.join(PROJ, 'results', 'nn_base_I2_v1', 'base_I2_kaggle_ids.csv'),
           'base_I1_I2': os.path.join(PROJ, 'results', 'nn_base_I1_I2_v1', 'base_I1_I2_kaggle_ids.csv'),
           'krr_v2': os.path.join(PROJ, 'results', 'krr_v2', 'krr_v2_kaggle_ids.csv')}.get(name)
    if ref and os.path.exists(ref):
        old = pd.read_csv(ref)[tab0.genes].to_numpy(); rec['max_abs_diff_vs_previous_export'] = float(np.abs(pred - old).max())
    info['files'][name] = rec
    print(f'{name:11s} ' + ' | '.join(f'{c[:3]} row RMS {r:.3f}' for c, r in rec['row_RMS'].items())
          + (f" | max |diff| vs previous export {rec['max_abs_diff_vs_previous_export']:.2e}" if 'max_abs_diff_vs_previous_export' in rec else ''), flush=True)
json.dump(info, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
print(f'DONE [t={time.time() - t0:.0f}s] -> {args.out}/blend_kaggle_ids.csv')
