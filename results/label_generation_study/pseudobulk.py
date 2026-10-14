"""Pseudobulk the TRAINING raw counts (test wells were removed by the organizers): sum counts per (plate, well, cell_type)."""
import time, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from scipy import sparse, io
t0 = time.time()
D = "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/data"; OUT = "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/results/pseudobulk_train"
meta = pd.read_csv(f"{D}/adata_obs_meta.csv")
meta["sample"] = meta.plate_name + "|" + meta.well + "|" + meta.cell_type
samples = meta.groupby("sample").agg(plate_name=("plate_name", "first"), well=("well", "first"), row=("row", "first"),
            library_id=("library_id", "first"), donor_id=("donor_id", "first"), cell_type=("cell_type", "first"),
            sm_name=("sm_name", "first"), control=("control", "first"), n_cells=("obs_id", "size")).reset_index()
assert meta.groupby("sample").sm_name.nunique().max() == 1
s_idx = {s: i for i, s in enumerate(samples["sample"])}
cell2sample = pd.Series(meta["sample"].map(s_idx).to_numpy(), index=meta.obs_id)
tab = pq.read_table(f"{D}/adata_train.parquet", columns=["obs_id", "gene", "count"], read_dictionary=["obs_id", "gene"])
tab = tab.unify_dictionaries().combine_chunks()
print("read", tab.num_rows, f"[t={time.time()-t0:.0f}s]", flush=True)
obs_col, gene_col = tab.column("obs_id").chunk(0), tab.column("gene").chunk(0)
obs_dict = obs_col.dictionary.to_numpy(zero_copy_only=False); genes = gene_col.dictionary.to_numpy(zero_copy_only=False)
obs_codes = obs_col.indices.to_numpy(); gene_codes = gene_col.indices.to_numpy()
cnt = tab.column("count").chunk(0).to_numpy().astype(np.int64)
samp_of_obsdict = cell2sample.reindex(obs_dict).to_numpy()
assert not np.isnan(samp_of_obsdict.astype(float)).any(), "cells missing in obs_meta"
rows = samp_of_obsdict.astype(np.int64)[obs_codes]
M = sparse.coo_matrix((cnt, (rows, gene_codes.astype(np.int64))), shape=(len(samples), len(genes))).tocsr()
M.sum_duplicates()
order = np.argsort(genes); genes = genes[order]; M = M[:, order]
print("pseudobulk", M.shape, "total counts", int(M.sum()), f"[t={time.time()-t0:.0f}s]", flush=True)
de = pd.read_parquet(f"{D}/de_train.parquet", columns=None)
de_genes = [c for c in de.columns if c not in ("cell_type", "sm_name", "sm_lincs_id", "SMILES", "control")]
print("genes in raw:", len(genes), "| in de_train:", len(de_genes), "| de_train genes present in raw:", len(set(de_genes) & set(genes)))
samples["lib_size"] = np.asarray(M.sum(1)).ravel()
io.mmwrite(f"{OUT}/counts.mtx", M.T.tocoo())          # genes x samples for R
pd.Series(genes).to_csv(f"{OUT}/genes.csv", index=False, header=["gene"])
samples.to_csv(f"{OUT}/samples.csv", index=False)
sparse.save_npz(f"{OUT}/counts_samples_x_genes.npz", M)
print(samples.groupby("cell_type").agg(n_samples=("sample", "size"), median_cells=("n_cells", "median")).to_string())
print(f"DONE [t={time.time()-t0:.0f}s]")
