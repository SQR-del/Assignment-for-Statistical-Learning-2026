import sys, numpy as np, pandas as pd
P = "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction"
de = pd.read_parquet(f"{P}/data/de_train.parquet"); G = [c for c in de.columns if c not in ("cell_type","sm_name","sm_lincs_id","SMILES","control")]
off = {(c, s): de.loc[i, G].to_numpy(np.float64) for i, (c, s) in enumerate(zip(de.cell_type, de.sm_name))}
def load(variant):
    d = f"{P}/results/pseudobulk_train/de_{variant}"; idx = pd.read_csv(f"{d}/index.csv"); out = {}
    for f, g in idx.groupby("file"):
        A = np.fromfile(f"{d}/{f}", dtype=np.float64).reshape(-1, len(G)).T if False else np.fromfile(f"{d}/{f}", dtype=np.float64).reshape(len(g), len(G))
        for (_, r), a in zip(g.sort_values("col").iterrows(), A): out[(r.cell_type, r.sm_name)] = a
    return out, idx
rows = []
for v in sys.argv[1:]:
    ours, idx = load(v)
    for k, a in ours.items():
        if k not in off: continue
        b = off[k]; ok = np.isfinite(a)
        rows.append(dict(variant=v, cell_type=k[0], sm_name=k[1], r=np.corrcoef(a[ok], b[ok])[0, 1], rmse=np.sqrt(np.mean((a[ok]-b[ok])**2)),
                         rms_ratio=np.sqrt(np.mean(a[ok]**2)) / np.sqrt(np.mean(b[ok]**2)), exact=np.mean(np.isclose(a[ok], b[ok], atol=1e-6))))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.groupby(["variant", "cell_type"])[["r", "rmse", "rms_ratio", "exact"]].median().round(4).to_string())
print(R.groupby("variant")[["r", "rmse", "rms_ratio", "exact"]].median().round(4).to_string())
