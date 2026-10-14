import numpy as np, pandas as pd
from scipy.special import ndtri_exp
P = "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction"; B = f"{P}/results/pseudobulk_train"
de = pd.read_parquet(f"{P}/data/de_train.parquet"); G = [c for c in de.columns if c not in ("cell_type","sm_name","sm_lincs_id","SMILES","control")]
off = {(c, s): de.loc[i, G].to_numpy(np.float64) for i, (c, s) in enumerate(zip(de.cell_type, de.sm_name))}
def load(d, ext=".bin"):
    idx = pd.read_csv(f"{d}/index.csv"); out = {}
    for f, g in idx.groupby("file"):
        A = np.fromfile(f"{d}/{f.replace('.bin', ext)}", dtype=np.float64).reshape(len(g), len(G))
        for (_, r), a in zip(g.sort_values("col").iterrows(), A): out[(r.cell_type, r.sm_name)] = a
    return out, idx
to_z = lambda y: np.sign(y) * -ndtri_exp(-np.abs(y) * np.log(10) - np.log(2))
rms = lambda x: np.sqrt(np.mean(np.square(x)))
full, fidx = load(f"{B}/de_lib_g18"); fullt, _ = load(f"{B}/de_lib_g18", ".t.bin")
rows = []; pairs = []
for ct in ("NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"):
    tag = "".join(ch if ch.isalnum() else "_" for ch in ct) + "_"
    d = f"{B}/sim_small_{tag}"; sm, sidx = load(d); smt, _ = load(d, ".t.bin")
    si = sidx[sidx.cell_type == ct].iloc[0]; fi = fidx[fidx.cell_type == ct].iloc[0]
    print(f"{ct}: small n={si.n_samples} df.resid={si.df_resid} df.prior={si.df_prior:.1f} | full n={fi.n_samples} df.resid={fi.df_resid} df.prior={fi.df_prior:.1f}")
    for (c, s), a in sm.items():
        if c != ct: continue
        b = full[(c, s)]; o = off.get((c, s)); ta, tb = smt[(c, s)], fullt[(c, s)]
        za, zb = to_z(a), to_z(b)
        rows.append(dict(ct=ct, sm=s, ctrl=bool(de.control[(de.cell_type == c) & (de.sm_name == s)].iloc[0]),
                         rms_small=rms(a), rms_full=rms(b), ratio=rms(b) / rms(a), zratio=rms(zb) / rms(za), tratio=rms(tb) / rms(ta),
                         slope=float(b @ a / (a @ a)), r_sf=np.corrcoef(a, b)[0, 1], r_t=np.corrcoef(ta, tb)[0, 1],
                         r_small_off=np.corrcoef(a, o)[0, 1], r_full_off=np.corrcoef(b, o)[0, 1],
                         rmse_small_off=rms(a - o), rmse_full_off=rms(b - o), rms_off=rms(o)))
        pairs.append((ct, s, a, b))
R = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
print("\nper cell type medians:"); print(R.groupby("ct")[["ratio", "zratio", "tratio", "slope", "r_sf", "r_t", "r_small_off", "r_full_off", "rmse_small_off", "rmse_full_off"]].median().round(3).to_string())
print("\ncorr(log ratio, log rms_small) =", np.corrcoef(np.log(R.ratio), np.log(R.rms_small))[0, 1].round(3))
# gene-level: E|full| given |small| bins, pooled
a = np.concatenate([p[2] for p in pairs]); b = np.concatenate([p[3] for p in pairs])
bins = [0, .5, 1, 2, 3, 5, 8, 12, 20, 400]; lab = pd.cut(np.abs(a), bins)
T = pd.DataFrame(dict(bin=lab, a=np.abs(a), b=np.abs(b), same=np.sign(a) == np.sign(b), bs=b * np.sign(a))).groupby("bin", observed=True).agg(n=("a", "size"), mean_small=("a", "mean"), mean_full_abs=("b", "mean"), mean_full_signed=("bs", "mean"), sign_agree=("same", "mean"))
print("\ngene-level pooled (|small| bins):"); print(T.round(3).to_string())
R.to_csv("/tmp/sqr_claude_diag/sim_compare_rows.csv", index=False)
