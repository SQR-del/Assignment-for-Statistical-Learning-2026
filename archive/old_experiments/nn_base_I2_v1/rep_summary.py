import numpy as np, pandas as pd, os
pd.set_option('display.width', 220)
runs = {'main(3 seeds)': 'abl_main', 'rep1': 'abl_rep1', 'rep2': 'abl_rep2', 'rep3': 'abl_rep3'}
def summ(g):
    a = g[g.proto == 'A'].groupby('name').err.mean()
    b = g[g.proto == 'V1'].copy(); b['target'] = b['name'].str.split('|').str[0]; b['weak'] = b['name'].str.endswith('|weak')
    s = dict(A15=a.drop('T cells CD8+').mean(), V1=b.groupby('target').err.mean().mean(), V1w=b[b.weak].groupby('target').err.mean().mean())
    s['objective'] = (s['V1w'] / 1.1625 + s['A15'] / 0.9894) / 2; return pd.Series(s)
tabs, rows = {}, {}
for lab, d in runs.items():
    f = f'/tmp/sqr_claude_diag/{d}/rows.csv'
    if not os.path.exists(f): print('missing', f); continue
    R = pd.read_csv(f); rows[lab] = R; tabs[lab] = R.groupby('variant').apply(summ)
obj = pd.DataFrame({k: v['objective'] for k, v in tabs.items()})
print('objective per run (lower is better):'); print(obj.round(4).to_string())
reps = [k for k in obj.columns if k.startswith('rep')]
print('\nimprovement vs base per replicate (negative = better):')
for v in obj.index:
    if v == 'base': continue
    d = obj.loc[v, reps] - obj.loc['base', reps]
    ok = (d < 0).all() and (-d.mean() > 2 * d.std(ddof=1))
    print(f'  {v:11s} ' + '  '.join(f'{x:+.4f}' for x in d) + f'   mean {d.mean():+.4f}  sd {d.std(ddof=1):.4f}   STABLE={ok}')
print('\nper-protocol means over replicates:')
print(pd.concat({k: tabs[k] for k in reps}).groupby(level=1).mean().round(4).to_string())
# pooled: average each row's error over the three replicates, then paired bootstrap vs base
P = pd.concat([rows[k].assign(rep=k) for k in reps]).groupby(['variant', 'proto', 'name', 'row']).err.mean().reset_index()
rng = np.random.default_rng(7)
print('\nreplicate-averaged paired bootstrap vs base:')
for v in [x for x in P.variant.unique() if x != 'base']:
    for lab, sel in (('V1w', lambda g: g[(g.proto == 'V1') & g['name'].str.endswith('|weak')]), ('V1 ', lambda g: g[g.proto == 'V1']),
                     ('A15', lambda g: g[(g.proto == 'A') & (g['name'] != 'T cells CD8+')])):
        x = sel(P[P.variant == v]).set_index(['name', 'row']).err; y = sel(P[P.variant == 'base']).set_index(['name', 'row']).err
        dd = (x - y.loc[x.index]).to_numpy(); bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
        print(f'  {v:11s} {lab} {dd.mean():+.4f}  95% CI [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]')
