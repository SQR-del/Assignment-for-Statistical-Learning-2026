"""Mechanistic label calibration: slp_small -> t (df_small) -> scale -> slp with df_full. Validate on T/NK sims (leave-one-cell-type-out)."""
import numpy as np, pandas as pd
from scipy import stats
exec(open('/tmp/sqr_claude_diag/sim_compare.py').read().split('full, fidx')[0])
full, fidx = load(f"{B}/de_lib_g18"); fullt, _ = load(f"{B}/de_lib_g18", ".t.bin")
TNK = ("NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells")
sm, smt, dfs, dff = {}, {}, {}, {}
for ct in TNK:
    tag = "".join(ch if ch.isalnum() else "_" for ch in ct) + "_"; d = f"{B}/sim_small_{tag}"
    a, idx = load(d); at, _ = load(d, ".t.bin")
    sm.update({k: v for k, v in a.items() if k[0] == ct}); smt.update({k: v for k, v in at.items() if k[0] == ct})
    r = idx[idx.cell_type == ct].iloc[0]; dfs[ct] = r.df_resid + r.df_prior
    r = fidx[fidx.cell_type == ct].iloc[0]; dff[ct] = r.df_resid + r.df_prior
for ct in ('B cells', 'Myeloid cells'):
    r = fidx[fidx.cell_type == ct].iloc[0]; dfs[ct] = r.df_resid + r.df_prior
print('df total (resid+prior): small', {k: round(v, 1) for k, v in dfs.items()}, '\nfull', {k: round(v, 1) for k, v in dff.items()})
def slp_from_t(t, df): return np.sign(t) * -(stats.t.logsf(np.abs(t), df) + np.log(2)) / np.log(10)
def t_from_slp(y, df): return np.sign(y) * stats.t.isf(np.minimum(10.0 ** -np.abs(y) / 2, 0.5), df)
def t_from_slp_log(y, df):   # stable for large |y|: solve logsf(t) = -|y| ln10 - ln2 by Newton from isf guess
    y = np.asarray(y, float); tgt = -np.abs(y) * np.log(10) - np.log(2)
    t = np.where(np.abs(y) < 250, stats.t.isf(np.clip(np.exp(tgt), 1e-300, 0.5), df), 40.0)
    for _ in range(30):
        f = stats.t.logsf(t, df) - tgt; g = -np.exp(stats.t.logpdf(t, df) - stats.t.logsf(t, df))
        t = np.maximum(t - f / g, 0)
    return np.sign(y) * t
# 1) check the inversion reproduces limma's own t
k = ("NK cells", "Alvocidib"); tt = t_from_slp_log(sm[k], dfs["NK cells"])
print('inversion check NK Alvocidib: max|t_inv - t_limma| =', np.abs(tt - smt[k]).max().round(4), '| max |slp| =', np.abs(sm[k]).max().round(1))
# 2) per-row t scaling full vs small
rows = []
for (ct, s), ts in smt.items():
    tf = fullt[(ct, s)]; rows.append(dict(ct=ct, sm=s, rms_t=rms(ts), c=float(tf @ ts / (ts @ ts)), c_rms=rms(tf) / rms(ts)))
Rt = pd.DataFrame(rows); print(Rt.groupby('ct')[['c', 'c_rms']].median().round(3).to_string())
fit = np.polyfit(np.log(Rt.rms_t), np.log(Rt.c), 1); print('pooled log c = %.3f + %.3f log rms_t' % (fit[1], fit[0]))
def calibrate(y, df_s, df_f, coef=None):
    t = t_from_slp_log(y, df_s)
    if coef is not None: t = t * np.exp(coef[1] + coef[0] * np.log(rms(t)))
    return slp_from_t(t, df_f)
# 3) leave-one-cell-type-out evaluation. anchors_sim = official + (small - full): what B/Mye-like anchors look like; target = official
res = []
for ct in TNK:
    others = Rt[Rt.ct != ct]; coef = np.polyfit(np.log(others.rms_t), np.log(others.c), 1)
    for (c, s), a in sm.items():
        if c != ct: continue
        o = off[(c, s)]; anc = o + a - full[(c, s)]
        v = dict(identity=anc, df_only=calibrate(anc, dfs[ct], dff[ct]), df_plus_scale=calibrate(anc, dfs[ct], dff[ct], coef))
        res.append(dict(ct=ct, sm=s, strong=rms(o) > 1.5, **{f'rmse_{n}': rms(x - o) for n, x in v.items()}, **{f'ratio_{n}': rms(x) / rms(o) for n, x in v.items()}))
E = pd.DataFrame(res); pd.set_option('display.width', 250)
print(E.groupby(['ct', 'strong'])[[c for c in E.columns if c.startswith(('rmse', 'ratio'))]].mean().round(3).to_string())
print('\nall rows mean:'); print(E[[c for c in E.columns if c.startswith(('rmse', 'ratio'))]].mean().round(4).to_string())
