import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/code')
from scipy import stats
from common import Table, rms
tab = Table(); Y = tab.Y
R0 = pickle.load(open('/tmp/sqr_claude_diag/v1full_none.pkl', 'rb')); R1 = pickle.load(open('/tmp/sqr_claude_diag/v1full_pre.pkl', 'rb'))
DFF, DFS = R0['dff'], R0['dfs']
def cal(y, ct): t = np.sign(y) * stats.t.isf(10.0 ** -np.abs(y) / 2, DFS[ct]); return np.sign(t) * -(stats.t.logsf(np.abs(t), DFF) + np.log(2)) / np.log(10)
cut = np.quantile([rms(Y[[tab.row_of[(c, d)] for c in tab.SRC]].mean(0)) for d in tab.QOFF], .9)
rows = []
for (v, t, d), p in R0['pred'].items():
    ct = tab.cells[t]; y = Y[tab.row_of[(t, d)]]; weak = rms(Y[[tab.row_of[(c, d)] for c in tab.SRC]].mean(0)) < cut
    cands = {v: p[0], v + '+post': cal(p[0], ct), v + '+pre': R1['pred'][(v, t, d)][0]}
    for name, pp in cands.items():
        rows.append(dict(model=name, ct=ct, sm=tab.comps[d], weak=weak, strong_truth=rms(y) > 1.5,
                         err_R=rms(pp - y), err_F=rms(pp - cal(y, ct)), pred_rms=rms(pp), truth_F_rms=rms(cal(y, ct)), truth_R_rms=rms(y)))
D = pd.DataFrame(rows)
def agg(g, col):
    return pd.Series(dict(V1=g.groupby('ct')[col].mean().mean(), V1w=g[g.weak].groupby('ct')[col].mean().mean(),
                          V1_strong=g[~g.weak].groupby('ct')[col].mean().mean()))
out = {}
for truth in ('R', 'F'):
    T = D.groupby('model').apply(lambda g: agg(g, f'err_{truth}'), include_groups=False)
    for c in T.columns: T[c + '/krr1'] = T[c] / T.loc['krr_v1', c]
    out[truth] = T
pd.set_option('display.width', 250)
order = ['krr_v1', 'base', 'base_I2', 'blend', 'krr_v2', 'base_I1_I2']
print('official private relative to KRR v1:  base 0.980 | base_I2 0.907 | blend 0.906\n')
for truth, lab in (('R', 'truth = official restricted-fit anchor (old V1)'), ('F', 'truth = anchor mapped to full-fit df (V1-full)')):
    print(lab); T = out[truth]
    print(T.loc[order, ['V1', 'V1w', 'V1_strong', 'V1/krr1', 'V1w/krr1', 'V1_strong/krr1']].round(3).to_string()); print()
print('candidate magnitude corrections under V1-full (truth F):')
T = out['F']; print(T.loc[[m for m in T.index if m.split('+')[0] in ('blend', 'base_I2', 'base_I1_I2', 'krr_v2')], ['V1', 'V1w', 'V1_strong', 'V1/krr1']].round(3).to_string())
print('\nmean pred RMS / truth RMS (strong rows):')
S = D[~D.weak].groupby('model')[['pred_rms', 'truth_R_rms', 'truth_F_rms']].mean(); print(S.loc[[m for m in S.index if m.split('+')[0] in order]].round(2).to_string())
D.to_csv('/tmp/sqr_claude_diag/v1full_rows.csv', index=False)
