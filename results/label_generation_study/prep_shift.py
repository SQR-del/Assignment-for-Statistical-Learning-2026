"""Collect restricted (small) and full per-cell-type fits for T/NK B/Mye-train compounds + df values -> shift_data.pkl"""
import numpy as np, pandas as pd, pickle
exec(open('/tmp/sqr_claude_diag/sim_compare.py').read().split('full, fidx')[0])
full, fidx = load(f"{B}/de_lib_g18")
D = dict(small={}, full={}, dfs={}, dff={}, r_full_off={})
for ct in ("NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"):
    tag = "".join(ch if ch.isalnum() else "_" for ch in ct) + "_"
    a, idx = load(f"{B}/sim_small_{tag}")
    r = idx[idx.cell_type == ct].iloc[0]; D['dfs'][ct] = float(r.df_resid + r.df_prior)
    r = fidx[fidx.cell_type == ct].iloc[0]; D['dff'][ct] = float(r.df_resid + r.df_prior)
    for k, v in a.items():
        if k[0] == ct: D['small'][k] = v; D['full'][k] = full[k]; D['r_full_off'][k] = float(np.corrcoef(full[k], off[k])[0, 1])
for ct in ('B cells', 'Myeloid cells'):
    r = fidx[fidx.cell_type == ct].iloc[0]; D['dfs'][ct] = float(r.df_resid + r.df_prior)
pickle.dump(D, open('/tmp/sqr_claude_diag/shift_data.pkl', 'wb'))
Y = de[G].to_numpy(); print('max |official| =', np.abs(Y).max(), '| any inf', np.isinf(Y).any(), '| rows', len(D['small']))
print('dfs', D['dfs']); print('dff', D['dff']); print('rows with r_full_off<0.8:', [k for k, v in D['r_full_off'].items() if v < 0.8])
