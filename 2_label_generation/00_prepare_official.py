"""00_prepare_official.py  按主办方 compute_de.ipynb 的步骤重建伪 bulk。
来源: https://github.com/openproblems-bio/neurips-2023-scripts
步骤:
  1. 读 adata_train.parquet，删掉 adata_excluded_ids.csv 中列出的 (obs_id, gene) 条目
  2. 按 plate_name + well + cell_type 把原始计数相加
输出 (2_label_generation/cache_official/):
  samples.csv, genes.txt, <细胞类型>.i32 (样本 x 基因, int32, 行优先), <细胞类型>_samples.csv, anchors.txt
用法: python3 00_prepare_official.py
"""
import os, time, zipfile
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy import sparse

t0 = time.time()
P = "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction"; D = P + "/data"; Q2 = P + "/2_label_generation"; OUT = Q2 + "/cache_official"
os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv(D + "/adata_obs_meta.csv")
tab = pq.read_table(D + "/adata_train.parquet", columns=["obs_id", "gene", "count"], read_dictionary=["obs_id", "gene"]).unify_dictionaries().combine_chunks()
oc, gc = tab.column("obs_id").chunk(0), tab.column("gene").chunk(0)
obs_dict = oc.dictionary.to_numpy(zero_copy_only=False); gene_dict = gc.dictionary.to_numpy(zero_copy_only=False)
o = oc.indices.to_numpy().astype(np.int64); g = gc.indices.to_numpy().astype(np.int64); cnt = tab.column("count").chunk(0).to_numpy().astype(np.int64)
print("adata_train entries", len(o), "cells", len(obs_dict), "genes", len(gene_dict), "[%.0fs]" % (time.time() - t0), flush=True)

zp = Q2 + "/adata_excluded_ids.csv.zip"
ex = pd.read_csv(zipfile.ZipFile(zp).open("adata_excluded_ids.csv")) if os.path.exists(zp) else pd.read_csv(D + "/adata_excluded_ids.csv")
print("excluded rows", len(ex), "columns", list(ex.columns), "| cells", ex.obs_id.nunique(), "genes", ex.gene.nunique(), flush=True)
om = {x: i for i, x in enumerate(obs_dict)}; gm = {x: i for i, x in enumerate(gene_dict)}
eo = ex.obs_id.map(om); eg = ex.gene.map(gm)
print("excluded entries whose cell / gene is not in adata_train:", int(eo.isna().sum()), int(eg.isna().sum()))
ok = eo.notna() & eg.notna(); NGD = len(gene_dict)
keep = ~np.isin(o * NGD + g, (eo[ok].to_numpy().astype(np.int64) * NGD + eg[ok].to_numpy().astype(np.int64)))
print("entries removed", int((~keep).sum()), "of", len(o), "| counts removed", int(cnt[~keep].sum()), "of", int(cnt.sum()), flush=True)
o, g, cnt = o[keep], g[keep], cnt[keep]

# 与笔记本一致：基因和细胞按（过滤后）首次出现的顺序排列
gu, gfirst = np.unique(g, return_index=True); gorder = gu[np.argsort(gfirst)]
gpos = np.full(NGD, -1); gpos[gorder] = np.arange(len(gorder)); genes = gene_dict[gorder]
meta = meta.set_index("obs_id").loc[obs_dict].reset_index()
meta["sample"] = meta.plate_name.astype(str) + "_" + meta.well.astype(str) + "_" + meta.cell_type.astype(str)
used = np.zeros(len(obs_dict), bool); used[np.unique(o)] = True
print("cells with at least one remaining entry:", int(used.sum()), "| genes remaining:", len(genes), flush=True)
samples = meta[used].groupby("sample", sort=True).agg(plate_name=("plate_name", "first"), well=("well", "first"), row=("row", "first"), library_id=("library_id", "first"),
            donor_id=("donor_id", "first"), cell_type=("cell_type", "first"), sm_name=("sm_name", "first"), control=("control", "first"), n_cells=("obs_id", "size")).reset_index()
sidx = {s: i for i, s in enumerate(samples["sample"])}
cell2s = meta["sample"].map(sidx).to_numpy()
M = sparse.coo_matrix((cnt, (cell2s[o].astype(np.int64), gpos[g])), shape=(len(samples), len(genes))).tocsr(); M.sum_duplicates()
samples["lib_size"] = np.asarray(M.sum(1)).ravel()
samples.to_csv(OUT + "/samples.csv", index=False); open(OUT + "/genes.txt", "w").write("\n".join(genes) + "\n")
sparse.save_npz(OUT + "/counts_samples_x_genes.npz", M)
tag = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s)
for ct, S in samples.groupby("cell_type"):
    M[S.index.to_numpy()].toarray().astype("<i4").tofile(OUT + "/" + tag(ct) + ".i32"); S.to_csv(OUT + "/" + tag(ct) + "_samples.csv", index=False)
    print("%-20s samples %4d  zero-count samples %d" % (ct, len(S), int((S.lib_size == 0).sum())))
b = samples[samples.cell_type == "B cells"].sm_name.unique()
open(OUT + "/anchors.txt", "w").write("\n".join(sorted(set(b) - {"Dimethyl Sulfoxide", "Belinostat", "Dabrafenib"})) + "\n")
de_genes = [c for c in pq.read_schema(D + "/de_train.parquet").names if c not in ("cell_type", "sm_name", "sm_lincs_id", "SMILES", "control")]
print("genes equal to de_train genes (as sets):", set(de_genes) == set(genes), "| same order:", de_genes == list(genes), "| DONE [%.0fs]" % (time.time() - t0))
