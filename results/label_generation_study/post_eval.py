import pickle, numpy as np, pandas as pd
from scipy import stats
P = "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction"
df = pd.read_parquet(P + '/data/de_train.parquet'); META = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
genes = [c for c in df.columns if c not in META]; Y = df[genes].to_numpy(np.float64)
SH = pickle.load(open('/tmp/sqr_claude_diag/shift_data.pkl', 'rb'))
rms = lambda x: np.sqrt(np.mean(np.square(x), -1))
def slp_from_t(t, d): return np.sign(t) * -(stats.t.logsf(np.abs(t), d) + np.log(2)) / np.log(10)
def t_from_slp(y, d): return np.sign(y) * stats.t.isf(10.0 ** -np.abs(y) / 2, d)      # |y| <= 180 here: no underflow
def dff_est(ct): return float(np.mean([v for k, v in SH['dff'].items() if k != ct]))
out = []
for run in ('abl2_none', 'abl2_sim', 'abl2_sim_pre'):
    PR = pickle.load(open(f'/tmp/sqr_claude_diag/{run}/preds.pkl', 'rb'))
    for (v, proto, name), (qry, pred) in PR.items():
        pred = pred.astype(np.float64); y = Y[qry]
        post = slp_from_t(t_from_slp(pred, SH['dfs'][name]), dff_est(name))
        s_opt = float((pred * y).sum() / (pred * pred).sum())
        row = dict(run=run, variant=v, target=name, err=rms(pred - y).mean(), err_post=rms(post - y).mean(), s_opt=s_opt)
        for s in (0.8, 0.9, 1.1, 1.2): row[f'err_x{s}'] = rms(s * pred - y).mean()
        out.append(row)
R = pd.DataFrame(out); R = R[R.target != 'T cells CD8+']
pd.set_option('display.width', 250)
T = R.groupby(['run', 'variant'])[['err', 'err_post', 'err_x0.8', 'err_x0.9', 'err_x1.1', 'err_x1.2', 's_opt']].mean()
print('A15 (NK, CD4, Treg) mean MRRMSE; err_post = df-calibrated predictions; err_xS = predictions scaled by S; s_opt = least-squares scale')
print(T.round(4).to_string())
