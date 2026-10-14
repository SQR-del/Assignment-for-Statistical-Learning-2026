"""Discriminate: were official B/Mye train labels fit on train wells only? Compare official 'identical-value' gene sets with genes of zero count in training wells."""
import numpy as np, pandas as pd, scipy.sparse as sp
P = "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction"; B = f"{P}/results/pseudobulk_train"
df = pd.read_parquet(P + '/data/de_train.parquet'); META = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
genes = [c for c in df.columns if c not in META]; Y = df[genes].to_numpy(np.float64)
S = pd.read_csv(f"{B}/samples.csv"); gall = pd.read_csv(f"{B}/genes.csv").gene.tolist()
C = sp.load_npz(f"{B}/counts_samples_x_genes.npz").tocsc()
gi = {g: i for i, g in enumerate(gall)}; cols = [gi[g] for g in genes]
C = C[:, cols].tocsr()
for ct in sorted(df.cell_type.unique()):
    rows = np.flatnonzero((df.cell_type == ct).to_numpy())
    # official: genes equal to row mode in ALL rows of the cell type
    core = None
    for i in rows:
        v, c = np.unique(np.round(Y[i], 9), return_counts=True); m = v[np.argmax(c)]
        s = set(np.flatnonzero(np.isclose(Y[i], m, atol=1e-9)))
        core = s if core is None else core & s
    sidx = np.flatnonzero((S.cell_type == ct).to_numpy())
    tot = np.asarray(C[sidx].sum(0)).ravel()
    zero = set(np.flatnonzero(tot == 0))
    low = {k: set(np.flatnonzero(tot <= k)) for k in (1, 2, 5)}
    print(f"{ct:20s} official constant genes {len(core):4d} | zero-count in train wells {len(zero):4d} | overlap {len(core & zero):4d} "
          f"| official-not-zero {len(core - zero):3d} | zero-not-official {len(zero - core):3d} | counts of zero-not-official genes: "
          f"{np.bincount(tot[list(zero - core)].astype(int)).tolist() if zero - core else '-'} | max count among official set {int(tot[list(core)].max()) if core else '-'}")
print("\nT/NK: genes with zero counts in the RESTRICTED sample set (17 B/Mye-train compounds + DMSO) -> how many are constant in the official (full-fit) labels?")
ONLY = set(df.sm_name[df.cell_type == 'B cells']) | {'Dimethyl Sulfoxide'}
for ct in ('NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells'):
    rows = np.flatnonzero((df.cell_type == ct).to_numpy()); core = None
    for i in rows:
        v, c = np.unique(np.round(Y[i], 9), return_counts=True); m = v[np.argmax(c)]
        s = set(np.flatnonzero(np.isclose(Y[i], m, atol=1e-9))); core = s if core is None else core & s
    sidx = np.flatnonzero(((S.cell_type == ct) & S.sm_name.isin(ONLY)).to_numpy())
    tot = np.asarray(C[sidx].sum(0)).ravel(); zr = set(np.flatnonzero(tot == 0))
    print(f"{ct:20s} restricted samples {len(sidx)} | zero in restricted {len(zr):4d} | of these constant in official {len(zr & core):4d} ({len(zr & core) / len(zr):.0%})")
