import numpy as np, pandas as pd
rows = []
for cfg in ('sim', 'sim_pre'):
    for rep, d in enumerate([f'abl2_{cfg}', f'abl2_{cfg}_r1', f'abl2_{cfg}_r2']):
        R = pd.read_csv(f'/tmp/sqr_claude_diag/{d}/rows.csv'); R = R[(R.proto == 'A') & (R.name != 'T cells CD8+')]
        R['cfg'] = cfg; R['rep'] = rep; rows.append(R)
R = pd.concat(rows)
A15 = R.groupby(['cfg', 'rep', 'variant', 'name']).err.mean().groupby(['cfg', 'rep', 'variant']).mean().unstack('cfg')
print('A15 per replicate (aligned protocol):'); print(A15.round(4).to_string())
print('\nmean over replicates:'); M = A15.groupby('variant').mean(); M['delta_pre'] = M.sim_pre - M.sim; print(M.round(4).to_string())
rng = np.random.default_rng(0)
for v in ('blend', 'base_I2', 'base_I1_I2', 'base', 'krr_v2', 'krr_v1'):
    a = R[(R.cfg == 'sim') & (R.variant == v)].groupby(['name', 'row']).err.mean(); b = R[(R.cfg == 'sim_pre') & (R.variant == v)].groupby(['name', 'row']).err.mean()
    d = (b - a.loc[b.index]).to_numpy(); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f'{v:11s} calib pre - none: {d.mean():+.4f}  95% CI [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  rows improved {np.mean(d < 0):.0%}')
