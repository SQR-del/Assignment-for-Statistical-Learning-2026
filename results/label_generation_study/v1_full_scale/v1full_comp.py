import numpy as np, pandas as pd
D = pd.read_csv('/tmp/sqr_claude_diag/v1full_rows.csv')
# test-like composite: by construction 10% of official test compounds lie above the source-strength cut (non-weak)
def comp(g, col):
    return 0.9 * g[g.weak].groupby('ct')[col].mean().mean() + 0.1 * g[~g.weak].groupby('ct')[col].mean().mean()
models = ['krr_v1', 'base', 'base_I2', 'blend', 'krr_v2', 'base_I1_I2', 'blend+post', 'blend+pre', 'base_I2+post', 'base_I2+pre', 'krr_v2+pre']
off = {'krr_v1': 1.0, 'base': 0.813 / 0.830, 'base_I2': 0.753 / 0.830, 'blend': 0.752 / 0.830}
T = pd.DataFrame({t: {m: comp(D[D.model == m], f'err_{t}') for m in models} for t in ('R', 'F')})
T['R/krr1'] = T.R / T.loc['krr_v1', 'R']; T['F/krr1'] = T.F / T.loc['krr_v1', 'F']; T['official/krr1'] = pd.Series(off)
print('test-like composite (0.9 * weak anchors + 0.1 * strong anchors), mean over B and Myeloid')
print(T.round(3).to_string())
rng = np.random.default_rng(0)
for a, b in (('blend+post', 'blend'), ('blend+pre', 'blend'), ('base_I2', 'base'), ('blend', 'krr_v1'), ('base', 'krr_v1')):
    x = D[D.model == a].set_index(['ct', 'sm']); y = D[D.model == b].set_index(['ct', 'sm']).loc[x.index]
    keys = x.index.get_level_values('sm').unique().to_numpy(); bs = []
    for _ in range(2000):
        s = rng.choice(keys, len(keys)); xs = pd.concat([x.xs(k, level='sm', drop_level=False) for k in s]); ys = pd.concat([y.xs(k, level='sm', drop_level=False) for k in s])
        if xs.weak.all() or (~xs.weak).all(): continue
        bs.append(comp(xs.reset_index(), 'err_F') / comp(ys.reset_index(), 'err_F') - 1)
    d = comp(x.reset_index(), 'err_F') / comp(y.reset_index(), 'err_F') - 1
    print(f'{a:12s} vs {b:7s}: {d:+.3f}  compound-bootstrap 95% CI [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]')
