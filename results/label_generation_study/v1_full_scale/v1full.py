"""B/Myeloid leave-one-anchor-out with training anchors kept at restricted-fit scale; saves fold predictions.
--pre: the visible B/Myeloid rows are df-recalibrated (restricted -> full) before training."""
import sys, os, time, pickle, argparse
sys.path.insert(0, '/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/code')
import numpy as np
from scipy import stats
from common import Table, rms
from krr import krr_predict
from nn import Config, prepare, ensemble_predict
p = argparse.ArgumentParser(); p.add_argument('--pre', action='store_true'); p.add_argument('--seeds', type=int, default=3)
p.add_argument('--seed', type=int, default=20260930); p.add_argument('--out', required=True)
args = p.parse_args(); t0 = time.time(); cfg = Config()
SH = pickle.load(open('/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/results/label_generation_study/shift_data.pkl', 'rb'))
DFF = float(np.mean(list(SH['dff'].values())))
def cal(y, ct): t = np.sign(y) * stats.t.isf(10.0 ** -np.abs(y) / 2, SH['dfs'][ct]); return np.sign(t) * -(stats.t.logsf(np.abs(t), DFF) + np.log(2)) / np.log(10)
tab = Table(); Y0 = tab.Y.copy()
PRED = {}
for t in tab.SPARSE:
    for d in tab.ANC:
        i = tab.row_of[(t, d)]; hid = np.array([i])
        tf = object.__new__(Table); tf.__dict__.update(tab.__dict__); tf.Y = Y0.copy()
        if args.pre:
            for j in np.flatnonzero(np.isin(tab.c_row, tab.SPARSE)):
                if j != i: tf.Y[j] = cal(Y0[j], tab.cells[tab.c_row[j]])
        S = prepare(tf, hid, cfg)
        for v in ('base', 'base_I2', 'base_I1_I2'):
            PRED[(v, t, d)], _ = ensemble_predict(S, tf, hid, v, args.seeds, args.seed, cfg)
        anc = [a for a in tab.ANC if a != d]
        for ver in (1, 2):
            PRED[(f'krr_v{ver}', t, d)] = krr_predict(tf.Y, tab.row_of, len(tab.comps), t, tab.SRC, anc, [d], ver, skip=set(tab.CTRLC))
        PRED[('blend', t, d)] = 0.4 * PRED[('base_I2', t, d)] + 0.4 * PRED[('base_I1_I2', t, d)] + 0.2 * PRED[('krr_v2', t, d)]
        print(f'{tab.cells[t]} | {tab.comps[d]} [t={time.time() - t0:.0f}s]', flush=True)
pickle.dump(dict(pred=PRED, dff=DFF, dfs={c: SH['dfs'][c] for c in ('B cells', 'Myeloid cells')}), open(args.out, 'wb'))
print('DONE')
