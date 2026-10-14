import numpy as np, pandas as pd
exec(open('/tmp/sqr_claude_diag/sim_compare.py').read().split('full, fidx')[0])
full, _ = load(f"{B}/de_lib_g18")
ANCH = sorted(set(de.sm_name[de.cell_type == 'B cells']))
small = {}
for ct in ("NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"):
    tag = "".join(ch if ch.isalnum() else "_" for ch in ct) + "_"
    s_, _ = load(f"{B}/sim_small_{tag}"); small.update({k: v for k, v in s_.items() if k[0] == ct})
bins = [0, .5, 1, 2, 3, 5, 8, 12, 20, 400]
def curve(pairs):
    a = np.concatenate([p[0] for p in pairs]); o = np.concatenate([p[1] for p in pairs])
    lab = pd.cut(np.abs(a), bins)
    return pd.DataFrame(dict(bin=lab, r=(o * np.sign(a)) / np.maximum(np.abs(a), 1e-9) , a=np.abs(a), o=o * np.sign(a))).groupby('bin', observed=True).apply(lambda g: pd.Series(dict(n=len(g), ratio=g.o.mean() / g.a.mean())))
out = {}
for grp, cts in (('NK+CD4', ("NK cells", "T cells CD4+")), ('CD8+Treg', ("T cells CD8+", "T regulatory cells"))):
    out[f'{grp} off/full'] = curve([(full[(c, s)], off[(c, s)]) for c in cts for s in ANCH if (c, s) in off])
    out[f'{grp} off/small'] = curve([(small[(c, s)], off[(c, s)]) for c in cts for s in ANCH if (c, s) in small])
for c in ('B cells', 'Myeloid cells'):
    out[f'{c[:3]} off/ours'] = curve([(full[(c, s)], off[(c, s)]) for s in ANCH if (c, s) in off])
T = pd.concat({k: v.ratio for k, v in out.items()}, axis=1); T['n_B'] = out['B c off/ours'].n
pd.set_option('display.width', 250); print(T.round(3).to_string())
# row-level: NK / CD4 per compound ratios for the strong ones
print()
for s in ('Alvocidib', 'Belinostat', 'MLN 2238', 'Oprozomib (ONX 0912)', 'Dactolisib', 'Dabrafenib'):
    line = f'{s[:22]:22s}'
    for c in ("NK cells", "T cells CD4+", "T regulatory cells"):
        if (c, s) in small: line += f' | {c[:6]} full/off {rms(full[(c,s)])/rms(off[(c,s)]):.2f} small/off {rms(small[(c,s)])/rms(off[(c,s)]):.2f}'
    for c in ('B cells', 'Myeloid cells'): line += f' | {c[:3]} ours/off {rms(full[(c,s)])/rms(off[(c,s)]):.2f}'
    print(line)
