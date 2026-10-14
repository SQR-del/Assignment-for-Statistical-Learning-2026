import numpy as np, pandas as pd
exec(open('/tmp/sqr_claude_diag/sim_compare.py').read().split('full, fidx')[0])
full, fidx = load(f"{B}/de_lib_g18")
R = pd.read_csv('/tmp/sqr_claude_diag/sim_compare_rows.csv')
rows = []
for (c, s), a in full.items():
    if (c, s) not in off: continue
    o = off[(c, s)]
    if c in ('B cells', 'Myeloid cells') or s in set(R.sm):
        rows.append(dict(ct=c, sm=s, rms_off=rms(o), ratio_ours_off=rms(a) / rms(o), slope_off_on_ours=float(o @ a / (a @ a)), r=np.corrcoef(a, o)[0, 1]))
X = pd.DataFrame(rows)
# T/NK: small-fit vs official ratio
small_ratio = {(r.ct, r.sm): r.rms_small / r.rms_off for r in R.itertuples()}
X['ratio_small_off'] = [small_ratio.get((c, s), np.nan) for c, s in zip(X.ct, X.sm)]
X['strong'] = X.rms_off > 1.5
pd.set_option('display.width', 250)
print(X[X.ct.isin(['B cells', 'Myeloid cells'])].sort_values(['ct', 'rms_off']).round(3).to_string(index=False))
print('\nsummary by group (strong = official row RMS > 1.5):')
X['grp'] = np.where(X.ct.isin(['B cells', 'Myeloid cells']), 'B/Mye ours(restricted) vs off', 'T/NK ours(full) vs off')
print(X.groupby(['grp', 'strong'])[['ratio_ours_off', 'slope_off_on_ours', 'r', 'ratio_small_off']].median().round(3).to_string())
