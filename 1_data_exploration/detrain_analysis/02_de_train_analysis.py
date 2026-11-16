# Copied from branch singlecell-perturbation-project (data_analysis/eda/, commit 536ed6e). Changes: `common` is imported
# as src/common_emb.py (a verbatim copy of that branch's code/common.py) and outputs go to ./outputs/.
"""Data-focused EDA of de_train.parquet (the label matrix used for training).

Produces all figures and tables in outputs/detrain/ and prints key numbers.
Run:  python 1_data_exploration/detrain_analysis/02_de_train_analysis.py
"""
import os
import sys
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
import common_emb as C

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9})

df = pd.read_parquet(C.data_path("de_train.parquet"))
genes = [c for c in df.columns if c not in
         ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]]
FULL = C.FULL_CELLS
ALL = C.ALL_CELLS

drug_df = df[~df.control].reset_index(drop=True)
by_cell = {c: drug_df[drug_df.cell_type == c].set_index("sm_name")[genes]
           for c in ALL}
shared6 = sorted(set.intersection(*[set(by_cell[c].index) for c in ALL]))
shared4 = sorted(set.intersection(*[set(by_cell[c].index) for c in FULL]))
anchors = shared6
test_drugs4 = [d for d in shared4 if d not in shared6]


def make_tensor(cells, drugs):
    return np.stack([by_cell[c].loc[drugs].to_numpy(np.float64) for c in cells])


T6 = make_tensor(ALL, shared6)
T4 = make_tensor(FULL, shared4)
print("T6", T6.shape, "T4", T4.shape, "shared-in-all-6", len(shared6),
      "shared4", len(shared4), "test-in-full4", len(test_drugs4))


def var_decomp(T):
    Cc, Dd, Gg = T.shape
    grand = T.mean(axis=(0, 1))
    cm = T.mean(axis=1)
    dm = T.mean(axis=0)
    sst = ((T - grand) ** 2).sum(axis=(0, 1))
    ssc = Dd * ((cm - grand) ** 2).sum(axis=0)
    ssd = Cc * ((dm - grand) ** 2).sum(axis=0)
    ssr = np.clip(sst - ssc - ssd, 0.0, None)
    return sst, ssc, ssd, ssr, cm, dm, grand


def decomp_table(T, tag):
    sst, ssc, ssd, ssr, cm, dm, grand = var_decomp(T)
    ok = sst > 1e-12
    frac = np.stack([ssc[ok] / sst[ok], ssd[ok] / sst[ok], ssr[ok] / sst[ok]], 1)
    rows = [{
        "design": tag, "n_cell": T.shape[0], "n_drug": T.shape[1],
        "component": name,
        "equal_gene_mean": float(frac[:, k].mean()),
        "variance_weighted": float([ssc, ssd, ssr][k].sum() / sst.sum()),
        "gene_q25": float(np.quantile(frac[:, k], 0.25)),
        "gene_median": float(np.median(frac[:, k])),
        "gene_q75": float(np.quantile(frac[:, k], 0.75)),
    } for k, name in enumerate(["cell", "drug", "interaction"])]
    per_gene = pd.DataFrame({
        "gene": np.array(genes)[ok],
        "ss_total": sst[ok], "frac_cell": frac[:, 0],
        "frac_drug": frac[:, 1], "frac_interaction": frac[:, 2],
        "cell_mean_abs": np.abs(cm)[:, ok].mean(0),
        "drug_mean_abs": np.abs(dm)[:, ok].mean(0),
    })
    return pd.DataFrame(rows), per_gene, (sst, ssc, ssd, ssr, cm, dm, grand)


vd6, pg6, parts6 = decomp_table(T6, "6x13_balanced")
vd4, pg4, parts4 = decomp_table(T4, "4x140_balanced")
vd = pd.concat([vd6, vd4], ignore_index=True)
vd.to_csv(os.path.join(OUT, "variance_components.csv"), index=False)
pg4.to_csv(os.path.join(OUT, "per_gene_variance.csv"), index=False)
print(vd.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
w = 0.2
x = np.arange(2)
for k, (comp, col) in enumerate(zip(["cell", "drug", "interaction"],
                                    ["#4C72B0", "#DD8452", "#55A868"])):
    sub = vd[vd.component == comp]
    ax[0].bar(x + (k - 1) * w, sub.variance_weighted, width=w, label=comp,
              color=col)
ax[0].set_xticks(x)
ax[0].set_xticklabels(["6 cells x 13 shared drugs", "4 cells x 140 drugs"])
ax[0].set_ylabel("share of total variance (variance-weighted)")
ax[0].set_title("Two-way variance decomposition")
ax[0].legend(frameon=False)
ax[1].hist(pg4.frac_interaction, bins=80, color="#55A868")
ax[1].set_xlabel("interaction fraction per gene")
ax[1].set_ylabel("# genes")
ax[1].set_title("Cell x drug interaction share (4 cells x 140 drugs)")
ax[1].axvline(pg4.frac_interaction.median(), color="k", ls="--", lw=1,
              label=f"median={pg4.frac_interaction.median():.2f}")
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "01_variance_components.png"))
plt.close(fig)

sst4, ssc4, ssd4, ssr4, cm4, dm4, grand4 = parts4
R4 = (T4 - cm4[:, None, :] - dm4[None, :, :] + grand4)
R6 = (T6 - parts6[4][:, None, :] - parts6[5][None, :, :] + parts6[6])


def rank_curve(mat, tag, n=120):
    mat = np.asarray(mat, np.float64)
    p = PCA(n_components=min(n, min(mat.shape) - 1), random_state=C.SEED)
    p.fit(mat)
    ev = np.cumsum(p.explained_variance_ratio_)
    return ev, tag


all602 = drug_df[genes].to_numpy(np.float64)
curves = [
    rank_curve((T4 - grand4).reshape(-1, len(genes)), "4x140 grand-centred"),
    rank_curve(R4.reshape(-1, len(genes)), "4x140 interaction residual"),
    rank_curve(T4.mean(0) - grand4, "140 drug signatures"),
    rank_curve(all602 - all602.mean(0), "602 drug rows (as in old EDA)"),
]
top_var_idx = np.argsort(T4.reshape(-1, len(genes)).var(0))[::-1][:5000]
Z4 = T4.reshape(-1, len(genes))[:, top_var_idx]
Z4 = (Z4 - Z4.mean(0)) / (Z4.std(0) + 1e-12)
curves.append(rank_curve(Z4, "standardised top-5000 genes"))
rank_rows = []
for ev, tag in curves:
    rank_rows.append({"view": tag, "pc": np.arange(1, len(ev) + 1),
                      "cum_var": ev})
rank_df = pd.concat([pd.DataFrame(r) for r in rank_rows], ignore_index=True)
rank_df.to_csv(os.path.join(OUT, "rank_curves.csv"), index=False)

fig, ax = plt.subplots(figsize=(6.5, 4))
for ev, tag in curves:
    ax.plot(np.arange(1, len(ev) + 1), ev, label=tag)
ax.axhline(0.90, color="k", ls=":", lw=1)
ax.set_xlim(1, 140)
ax.set_ylim(0, 1.02)
ax.set_xlabel("# principal components")
ax.set_ylabel("cumulative variance explained")
ax.set_title("Rank of the label matrix (4 cells x 140 drugs)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_rank_curves.png"))
plt.close(fig)
for ev, tag in curves:
    r90 = int(np.searchsorted(ev, 0.90) + 1)
    r95 = int(np.searchsorted(ev, 0.95) + 1)
    print(f"rank {tag}: pc1={ev[0]:.3f} r90={r90} r95={r95}")

pca4 = PCA(n_components=12, random_state=C.SEED).fit(Z4)
scores4 = pca4.transform(Z4).reshape(T4.shape[0], T4.shape[1], -1)
load4 = pca4.components_
zload = (load4 - load4.mean(1, keepdims=True)) / (load4.std(1, keepdims=True) + 1e-12)
sel_genes = np.array(genes)[top_var_idx]

lrows = []
for k in range(12):
    order = np.argsort(load4[k])
    for side, idx in [("low", order[:15]), ("high", order[::-1][:15])]:
        for rank, j in enumerate(idx):
            lrows.append({"pc": k + 1, "side": side, "rank": rank + 1,
                          "gene": sel_genes[j], "loading": load4[k, j]})
pd.DataFrame(lrows).to_csv(os.path.join(OUT, "pc_loadings_top.csv"),
                           index=False)

markers = {
    "interferon": "ISG15 IFIT1 IFIT2 IFIT3 MX1 MX2 OAS1 OAS2 OAS3 STAT1 IRF7 IFI6 IFI44L XAF1".split(),
    "cell_cycle": "MKI67 TOP2A CDK1 CCNB1 CCNB2 BUB1 AURKA AURKB PLK1 CDC20 PCNA".split(),
    "apoptosis": "BAX BAK1 BCL2 BCL2L1 CASP3 CASP8 CASP9 PMAIP1 BBC3 FAS".split(),
    "heat_shock": "HSPA1A HSPA1B HSPA6 HSP90AA1 HSP90AB1 DNAJB1 HSPH1".split(),
    "AP1_immediate": "JUN JUNB FOS FOSB EGR1 ATF3 IER2 IER3".split(),
    "MHC": "HLA-A HLA-B HLA-C HLA-DRA HLA-DRB1 HLA-DPA1 HLA-DPB1 B2M TAP1".split(),
    "ribosome": "RPL3 RPL4 RPL5 RPL6 RPL7 RPL8 RPL10 RPS3 RPS4X RPS5 RPS6 RPS8 RPS18".split(),
    "MYC_growth": "MYC MYCN CCND1 E2F1 CDK4 CDK6 RRM2 TYMS".split(),
}
gene_pos = {g: i for i, g in enumerate(sel_genes)}
enr = []
for name, gs in markers.items():
    idx = [gene_pos[g] for g in gs if g in gene_pos]
    if not idx:
        continue
    for k in range(12):
        enr.append({"marker": name, "pc": k + 1,
                    "mean_z_loading": float(zload[k, idx].mean()),
                    "n_genes": len(idx)})
enr_df = pd.DataFrame(enr)
enr_df.to_csv(os.path.join(OUT, "marker_enrichment.csv"), index=False)
mat = enr_df.pivot(index="marker", columns="pc", values="mean_z_loading")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5),
                       gridspec_kw={"width_ratios": [1.1, 1]})
im = ax[0].imshow(mat.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax[0].set_yticks(range(mat.shape[0]))
ax[0].set_yticklabels(mat.index)
ax[0].set_xticks(range(mat.shape[1]))
ax[0].set_xticklabels([f"PC{c}" for c in mat.columns], rotation=45)
ax[0].set_title("Marker-set mean loading (z-scored)")
fig.colorbar(im, ax=ax[0], fraction=0.03)
top = pd.DataFrame(lrows)
sub1 = top[(top.pc == 1) & (top.side == "high")].head(10)
sub2 = top[(top.pc == 2) & (top.side == "high")].head(10)
y1 = np.arange(len(sub1)) * 2 + 0.4
y2 = y1 + 0.9
ax[1].barh(y1, sub1.loading, height=0.8, color="#C44E52", label="PC1 top+")
ax[1].barh(y2, sub2.loading, height=0.8, color="#4C72B0", label="PC2 top+")
ax[1].set_yticks(np.concatenate([y1, y2]))
ax[1].set_yticklabels(list(sub1.gene) + list(sub2.gene), fontsize=7)
ax[1].invert_yaxis()
ax[1].set_xlabel("loading")
ax[1].set_title("Top positive genes (PC1 red, PC2 blue)")
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_gene_modules.png"))
plt.close(fig)

cell_pc = np.stack([scores4[k].mean(0) for k in range(4)])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
im = ax[0].imshow(cell_pc[:, :8], cmap="RdBu_r", aspect="auto",
                  vmin=-np.abs(cell_pc[:, :8]).max(),
                  vmax=np.abs(cell_pc[:, :8]).max())
ax[0].set_yticks(range(4))
ax[0].set_yticklabels(FULL, fontsize=8)
ax[0].set_xticks(range(8))
ax[0].set_xticklabels([f"PC{i+1}" for i in range(8)])
ax[0].set_title("Mean PC score by cell type")
fig.colorbar(im, ax=ax[0], fraction=0.03)
pc1 = scores4[:, :, 0]
pc2 = scores4[:, :, 1]
for k, c in enumerate(FULL):
    ax[1].scatter(pc1[k], pc2[k], s=12, alpha=0.7, label=c)
lo1, hi1 = np.quantile(pc1, [0.005, 0.995])
lo2, hi2 = np.quantile(pc2, [0.005, 0.995])
n_out = int(((pc1 < lo1) | (pc1 > hi1) | (pc2 < lo2) | (pc2 > hi2)).sum())
ax[1].set_xlim(lo1, hi1)
ax[1].set_ylim(lo2, hi2)
ax[1].text(0.02, 0.98, f"{n_out} extreme points clipped", fontsize=7,
           transform=ax[1].transAxes, va="top")
ax[1].set_xlabel("PC1 score")
ax[1].set_ylabel("PC2 score")
ax[1].set_title("Per-drug scores (4 cells x 140 drugs)")
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_pc_scores.png"))
plt.close(fig)

sig4 = R4.transpose(1, 0, 2).reshape(len(shared4), -1)
resp_corr = np.corrcoef(sig4)
fp = C.smiles_fingerprints(
    pd.Series([drug_df[drug_df.sm_name == d].SMILES.iloc[0] for d in shared4]))
chem = normalize(fp) @ normalize(fp).T

iu = np.triu_indices(len(shared4), 1)
obs_mantel = float(np.corrcoef(resp_corr[iu], chem[iu])[0, 1])
rng = np.random.default_rng(C.SEED)
null = []
for _ in range(2000):
    p = rng.permutation(len(shared4))
    null.append(float(np.corrcoef(resp_corr[np.ix_(p, p)][iu], chem[iu])[0, 1]))
null = np.array(null)
pval = float((np.sum(null >= obs_mantel) + 1) / (len(null) + 1))
print(f"Mantel response~chemical r={obs_mantel:.3f} p={pval:.4f}")

pairs = []
for i, j in itertools.combinations(range(len(shared4)), 2):
    pairs.append({"drug_a": shared4[i], "drug_b": shared4[j],
                  "resp_corr": resp_corr[i, j], "chem_cos": chem[i, j]})
pairs = pd.DataFrame(pairs).sort_values("resp_corr", ascending=False)
pairs.to_csv(os.path.join(OUT, "drug_similarity_pairs.csv"), index=False)

dist = 1 - resp_corr
np.fill_diagonal(dist, 0.0)
dist = (dist + dist.T) / 2
Zlink = linkage(squareform(dist, checks=False), method="average")
leaves = dendrogram(Zlink, no_plot=True)["leaves"]
cl = fcluster(Zlink, t=10, criterion="maxclust")
drug_stats = pd.DataFrame({
    "drug": shared4, "is_anchor": [d in shared6 for d in shared4],
    "cluster": cl,
    "mean_abs": np.abs(T4).mean(axis=(0, 2)),
    "rms": np.sqrt((T4 ** 2).mean(axis=(0, 2))),
    "interaction_rms": np.sqrt((R4 ** 2).mean(axis=(0, 2))),
})
cc = []
for i in range(len(shared4)):
    rs = [np.corrcoef(T4[a, i], T4[b, i])[0, 1]
          for a, b in itertools.combinations(range(4), 2)]
    cc.append(float(np.nanmean(rs)))
drug_stats["cross_cell_r"] = cc
drug_stats["chem_to_anchor_max"] = [
    max(chem[i, [shared4.index(a) for a in shared6]]) for i in range(len(shared4))]
drug_stats.to_csv(os.path.join(OUT, "drug_stats.csv"), index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 5),
                       gridspec_kw={"width_ratios": [1.25, 1]})
ordr = np.array(leaves)
im = ax[0].imshow(resp_corr[np.ix_(ordr, ordr)], cmap="viridis",
                  vmin=np.quantile(resp_corr, 0.01), vmax=np.quantile(resp_corr, 0.99))
anch_pos = [list(ordr).index(shared4.index(a)) for a in shared6]
for p in anch_pos:
    ax[0].axhline(p - 0.5, color="red", lw=0.5)
    ax[0].axvline(p - 0.5, color="red", lw=0.5)
ax[0].set_title(f"Drug response similarity (ordered); red = {len(shared6)} anchors")
ax[0].set_xlabel("drug (clustered)")
ax[0].set_ylabel("drug (clustered)")
fig.colorbar(im, ax=ax[0], fraction=0.04)
ax[1].scatter(pairs.chem_cos, pairs.resp_corr, s=4, alpha=0.3)
ax[1].set_xlabel("SMILES char n-gram cosine")
ax[1].set_ylabel("response signature Pearson r")
ax[1].set_title(f"Mantel r={obs_mantel:.3f}, p={pval:.4f}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "05_drug_similarity.png"))
plt.close(fig)

anch = drug_stats[drug_stats.is_anchor]
test = drug_stats[~drug_stats.is_anchor]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for k, col in enumerate(["mean_abs", "cross_cell_r", "interaction_rms"]):
    ax[k].boxplot([anch[col].values, test[col].values],
                  tick_labels=[f"{len(anch)} anchors", f"{len(test)} others"])
    u = stats.mannwhitneyu(anch[col], test[col], alternative="two-sided")
    ax[k].set_title(f"{col}\nMann-Whitney p={u.pvalue:.3f}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "06_anchor_vs_test.png"))
plt.close(fig)

sig_z = (sig4 - sig4.mean(0)) / (sig4.std(0) + 1e-12)
pcsig = PCA(n_components=20, random_state=C.SEED).fit_transform(sig_z)
anchor_idx = [shared4.index(a) for a in shared6]
test_idx = [i for i in range(len(shared4)) if i not in anchor_idx]
d_obs = np.array([np.min(np.linalg.norm(pcsig[t] - pcsig[anchor_idx], axis=1))
                  for t in test_idx]).mean()
rand = []
for _ in range(2000):
    a = rng.choice(len(shared4), len(shared6), replace=False)
    rest = [i for i in range(len(shared4)) if i not in a]
    rand.append(np.mean([np.min(np.linalg.norm(pcsig[t] - pcsig[a], axis=1))
                         for t in rest]))
rand = np.array(rand)
cover_pct = float((rand <= d_obs).mean())
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(rand, bins=40, color="0.7", label=f"random {len(shared6)}-drug sets")
ax.axvline(d_obs, color="red", lw=2,
           label=f"actual anchors={d_obs:.1f} (pct {cover_pct:.2f})")
ax.set_xlabel("mean distance from non-anchor drugs to nearest anchor (PC space)")
ax.set_ylabel("# random sets")
ax.set_title(f"Do the {len(shared6)} anchors cover the drug-response space?")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "07_anchor_coverage.png"))
plt.close(fig)
print(f"anchor coverage distance={d_obs:.3f} percentile={cover_pct:.3f}")

b_my = {}
for c in ["B cells", "Myeloid cells"]:
    b_my[c] = by_cell[c].loc[shared6][genes].to_numpy(np.float64)
full6 = np.stack([by_cell[c].loc[shared6][genes].to_numpy(np.float64) for c in FULL])
full6_mean = full6.mean(0)
rows = []
for c in ["B cells", "Myeloid cells"]:
    for i, d in enumerate(shared6):
        rows.append({
            "cell": c, "drug": d,
            "corr_to_full_mean": float(np.corrcoef(b_my[c][i], full6_mean[i])[0, 1]),
            "mean_abs": float(np.abs(b_my[c][i]).mean()),
            "full_mean_abs": float(np.abs(full6_mean[i]).mean()),
        })
bm = pd.DataFrame(rows)
bm.to_csv(os.path.join(OUT, "anchor_transfer.csv"), index=False)
print(bm.groupby("cell")[["corr_to_full_mean", "mean_abs", "full_mean_abs"]].mean().to_string())
print("B vs Myeloid anchor corr:",
      float(np.mean([np.corrcoef(b_my["B cells"][i], b_my["Myeloid cells"][i])[0, 1]
                     for i in range(len(shared6))])))

allY = T4.reshape(-1, len(genes))
flat = allY.ravel()
q = [0, 1, 5, 25, 50, 75, 90, 95, 99, 99.9, 100]
quant = pd.DataFrame({"q": q, "abs_value": [float(np.percentile(np.abs(flat), x)) for x in q]})
quant["value"] = [float(np.percentile(flat, x)) for x in q]
quant.to_csv(os.path.join(OUT, "label_quantiles.csv"), index=False)
flat_all = drug_df[genes].to_numpy(np.float64).ravel()
frac_zero = float((flat_all == 0).mean())
frac_pos = float((flat_all > 0).mean())
frac_neg = float((flat_all < 0).mean())
frac_lt1 = float((np.abs(flat_all) < 1).mean())
print(f"sign(all 602): +{frac_pos:.3f} -{frac_neg:.3f} zero {frac_zero:.4f}; "
      f"|y|<1 {frac_lt1:.3f}; balanced4 {float((np.abs(flat) < 1).mean()):.3f}")

gmean = allY.mean(0)
gvar = allY.var(0)
gbar = np.abs(allY).mean(0)
sse_g = (allY - gmean) ** 2
sse_gene = sse_g.sum(0)
order = np.argsort(sse_gene)[::-1]
cum = np.cumsum(sse_gene[order]) / sse_gene.sum()
shares = {f"top{pct}%": float(cum[int(len(cum) * pct / 100) - 1])
          for pct in [1, 5, 10, 20, 50]}
print("SSE share of global-mean residual:", shares)

per_row_rmse = np.sqrt(((allY - gmean) ** 2).mean(1))
row_df = pd.DataFrame({
    "cell_type": np.repeat(FULL, T4.shape[1]),
    "drug": shared4 * 4,
    "mean_abs": np.abs(allY).mean(1),
    "rms": np.sqrt((allY ** 2).mean(1)),
    "rmse_to_grandmean": per_row_rmse,
})
row_df.to_csv(os.path.join(OUT, "per_row_stats.csv"), index=False)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].hist(np.abs(flat), bins=200, color="#4C72B0")
ax[0].set_yscale("log")
ax[0].set_xlabel("|signed -log10 p|")
ax[0].set_ylabel("count (log)")
ax[0].set_title("Label magnitude")
ax[1].scatter(gbar, np.sqrt(gvar), s=4, alpha=0.3)
ax[1].set_xlabel("gene mean |value|")
ax[1].set_ylabel("gene sd")
ax[1].set_title("Gene mean-variance relation")
ax[2].plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum,
           color="#C44E52", label="genes")
row_sse_fig = ((T4 - parts4[6]) ** 2).sum(axis=2).ravel()
row_cum_fig = np.cumsum(np.sort(row_sse_fig)[::-1]) / row_sse_fig.sum()
ax[2].plot(np.arange(1, len(row_cum_fig) + 1) / len(row_cum_fig) * 100,
           row_cum_fig, color="#4C72B0", label="rows")
ax[2].axhline(0.5, color="k", ls=":")
ax[2].set_xlabel("% genes or % rows (sorted by SSE contribution)")
ax[2].set_ylabel("cumulative SSE share")
ax[2].set_title("SSE concentration over genes and rows")
ax[2].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "08_label_structure.png"))
plt.close(fig)

grand = parts4[6]
anchor_pos = [shared4.index(d) for d in shared6]
grand_o = np.zeros_like(T4)
pred_transfer = np.zeros_like(T4)
csig = np.zeros_like(T4)
for k in range(4):
    others = [j for j in range(4) if j != k]
    cm_o = T4[others].mean(axis=0)
    grand_o[k] = cm_o.mean(axis=0)
    pred_transfer[k] = cm_o
    s = T4[k, anchor_pos].sum(axis=0)
    csig[k] = s / len(anchor_pos)
    for j in anchor_pos:
        csig[k, j] = (s - T4[k, j]) / (len(anchor_pos) - 1)

row_rmse_transfer = np.concatenate(
    [np.sqrt(((T4[k] - pred_transfer[k]) ** 2).mean(1)) for k in range(4)])

cands = {
    "global_mean(train cells)": grand_o,
    "dsig(drug mean of others)": pred_transfer,
    "csig(anchor mean target)": csig,
    "additive(csig+dsig-grand)": csig + pred_transfer - grand_o,
}
base_rows = []
yt = T4.reshape(-1, len(genes))
for name, P in cands.items():
    m, _ = C.compute_metrics(yt, P.reshape(-1, len(genes)), genes)
    base_rows.append({"model": name, **m})
base = pd.DataFrame(base_rows)
base.to_csv(os.path.join(OUT, "candidate_predictors.csv"), index=False)
print(base[["model", "mrrmse", "mse", "r2", "pearson_mean"]].to_string(index=False))

alphas = np.linspace(0, 1.6, 321)
curve = []
for a in alphas:
    P = grand_o + a * (pred_transfer - grand_o)
    m, _ = C.compute_metrics(yt, P.reshape(-1, len(genes)), genes)
    curve.append({"alpha": a, "mrrmse": m["mrrmse"], "mse": m["mse"],
                  "r2": m["r2"]})
curve = pd.DataFrame(curve)
curve.to_csv(os.path.join(OUT, "alpha_curve.csv"), index=False)
a_mrr = float(curve.loc[curve.mrrmse.idxmin(), "alpha"])
a_mse = float(curve.loc[curve.mse.idxmin(), "alpha"])
a_r2 = float(curve.loc[curve.r2.idxmax(), "alpha"])
print(f"optimal alpha: mrrmse={a_mrr:.3f} mse={a_mse:.3f} r2={a_r2:.3f}")

row_sse = ((T4 - grand_o) ** 2).sum(axis=2).ravel()
row_cum = np.cumsum(np.sort(row_sse)[::-1]) / row_sse.sum()
row_shares = {f"top{p}%": float(row_cum[int(len(row_cum) * p / 100) - 1])
              for p in [1, 5, 10, 20, 50]}
print("row SSE share of global-mean residual:", row_shares)

rms_rows = np.sqrt((T4 ** 2).mean(2))
top_mask = np.zeros((4, len(shared4)), bool)
for k in range(4):
    top_mask[k, np.argsort(rms_rows[k])[::-1][:14]] = True
rng2 = np.random.default_rng(C.SEED)
eA = np.where(top_mask[..., None], rng2.standard_normal(T4.shape), 0.0)
eB = rng2.standard_normal(T4.shape)
sim = []
for name, e in [("error on 10% hardest rows", eA),
                ("error spread over all rows", eB)]:
    e = e - e.mean()
    e = e * np.sqrt(2.0 / (e ** 2).mean())
    m, _ = C.compute_metrics(yt, (T4 + e).reshape(-1, len(genes)), genes)
    sim.append({"scenario": name, **m})
sim = pd.DataFrame(sim)
sim.to_csv(os.path.join(OUT, "metric_geometry.csv"), index=False)
print(sim[["scenario", "mrrmse", "mse", "r2"]].to_string(index=False))
with open(os.path.join(OUT, "row_sse_share.json"), "w", encoding="utf-8") as f:
    json.dump(row_shares, f, indent=2)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(curve.alpha, curve.mrrmse, color="#4C72B0", label="mrrmse")
ax2 = ax[0].twinx()
ax2.plot(curve.alpha, curve.mse, color="#C44E52", label="MSE")
ax[0].axvline(a_mrr, color="#4C72B0", ls="--", lw=1,
              label=f"mrrmse-opt a={a_mrr:.2f}")
ax2.axvline(a_mse, color="#C44E52", ls="--", lw=1,
            label=f"MSE-opt a={a_mse:.2f}")
ax[0].set_xlabel("shrinkage alpha on drug effect")
ax[0].set_ylabel("mrrmse", color="#4C72B0")
ax2.set_ylabel("MSE", color="#C44E52")
ax[0].set_title("Shrinkage curve of honest dsig")
lines = ax[0].get_lines() + ax2.get_lines()
ax[0].legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=7)
ax[1].scatter(base.mrrmse, base.r2, s=40)
for _, r in base.iterrows():
    ax[1].annotate(r.model, (r.mrrmse, r.r2), fontsize=7,
                   xytext=(3, 3), textcoords="offset points")
ax[1].set_xlabel("mrrmse (lower better)")
ax[1].set_ylabel("R2 (higher better)")
ax[1].set_title("Honest baselines: metric rankings")
ax[2].bar(np.arange(2), sim.mrrmse, color=["#C44E52", "#4C72B0"], width=0.5)
ax[2].set_xticks(np.arange(2))
ax[2].set_xticklabels(["concentrated\non 10% rows", "spread over\nall rows"])
ax[2].set_ylabel("mrrmse")
ax[2].set_title(f"Same MSE ({sim.mse.iloc[0]:.2f}) but different mrrmse")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "09_metric_conflict.png"))
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, c in enumerate(FULL):
    ax[0].scatter(np.abs(T4[k]).mean(1),
                  np.sqrt(((T4[k] - pred_transfer[k]) ** 2).mean(1)),
                  s=10, alpha=0.6, label=c)
ax[0].set_xlabel("row mean |value| (target cell)")
ax[0].set_ylabel("row RMSE of transfer baseline")
ax[0].set_title("Transfer error vs response strength")
ax[0].legend(frameon=False, fontsize=7)
ax[1].boxplot([np.sqrt(((T4[k] - pred_transfer[k]) ** 2).mean(1)) for k in range(4)],
              tick_labels=[c.replace("T cells ", "") for c in FULL])
ax[1].set_ylabel("per-row RMSE")
ax[1].set_title("Transfer baseline by held-out cell type")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "10_transfer_baseline.png"))
plt.close(fig)

rel = pg4[["gene", "frac_interaction", "drug_mean_abs"]].copy()
rel["consistency"] = 1 - rel.frac_interaction
print("gene consistency vs mean|y| spearman:",
      stats.spearmanr(rel.consistency, rel.drug_mean_abs))
print("drug cross_cell_r vs mean_abs spearman:",
      stats.spearmanr(drug_stats.cross_cell_r, drug_stats.mean_abs))

ctrl = df[df.control].reset_index(drop=True)
ctrl_rows = []
for _, r in ctrl.iterrows():
    v = r[genes].to_numpy(np.float64)
    ctrl_rows.append({"cell_type": r.cell_type, "drug": r.sm_name,
                      "mean_abs": float(np.abs(v).mean()),
                      "rms": float(np.sqrt((v ** 2).mean()))})
ctrl_df = pd.DataFrame(ctrl_rows)
for d in ctrl_df.drug.unique():
    M = ctrl_df[ctrl_df.drug == d].set_index("cell_type")
    rs = [np.corrcoef(ctrl.loc[ctrl.sm_name == d].set_index("cell_type").loc[a, genes].to_numpy(float),
                      ctrl.loc[ctrl.sm_name == d].set_index("cell_type").loc[b, genes].to_numpy(float))[0, 1]
          for a, b in itertools.combinations(M.index, 2)]
    ctrl_df.loc[ctrl_df.drug == d, "cross_cell_r_mean"] = float(np.nanmean(rs))
ctrl_df.to_csv(os.path.join(OUT, "control_rows.csv"), index=False)

transfer_rows = []
for k, c in enumerate(FULL):
    for name, P in [("global_mean", grand_o[k]), ("dsig", pred_transfer[k]),
                    ("csig", csig[k])]:
        m, _ = C.compute_metrics(T4[k], np.asarray(P), genes)
        transfer_rows.append({"cell_type": c, "baseline": name,
                              "mrrmse": m["mrrmse"], "mse": m["mse"],
                              "r2": m["r2"]})
transfer_df = pd.DataFrame(transfer_rows)
transfer_df.to_csv(os.path.join(OUT, "transfer_by_cell.csv"), index=False)

out_rows = []
for k, c in enumerate(FULL):
    for j, d in enumerate(shared4):
        out_rows.append({"cell_type": c, "drug": d,
                         "pc1": float(scores4[k, j, 0]),
                         "pc2": float(scores4[k, j, 1])})
pd.DataFrame(out_rows).sort_values("pc1", key=np.abs, ascending=False) \
    .head(20).to_csv(os.path.join(OUT, "pc_outliers.csv"), index=False)

b_only = sorted(set(by_cell["B cells"].index) - set(by_cell["Myeloid cells"].index))
m_only = sorted(set(by_cell["Myeloid cells"].index) - set(by_cell["B cells"].index))
cd8_missing = sorted(set(by_cell["T cells CD4+"].index) - set(by_cell["T cells CD8+"].index))
design = {
    "rows_total": int(len(df)), "rows_drug": int(len(drug_df)),
    "rows_control": int(len(ctrl)), "genes": len(genes),
    "cell_types": ALL, "n_drugs": int(drug_df.sm_name.nunique()),
    "drugs_per_cell": {c: int(by_cell[c].shape[0]) for c in ALL},
    "anchors_B_Myeloid": sorted(set(by_cell["B cells"].index) & set(by_cell["Myeloid cells"].index)),
    "B_only": b_only, "Myeloid_only": m_only,
    "CD8_missing": cd8_missing,
    "control_drugs": sorted(ctrl.sm_name.unique()),
    "shared_all6": len(shared6), "shared_all4": len(shared4),
    "test_drugs_total": int(drug_df.sm_name.nunique()) - 15,
    "test_drugs_in_balanced4": len(test_drugs4),
}
with open(os.path.join(OUT, "design_summary.json"), "w", encoding="utf-8") as f:
    json.dump(design, f, indent=2, ensure_ascii=False)

summary = {
    "n_rows_total": int(len(df)), "n_rows_drug": int(len(drug_df)),
    "n_genes": len(genes), "n_cells": len(ALL), "n_drugs": int(drug_df.sm_name.nunique()),
    "shared6": len(shared6), "shared4": len(shared4),
    "frac_abs_lt1_all602": frac_lt1, "frac_pos": frac_pos, "frac_neg": frac_neg,
    "frac_abs_lt1_balanced4": float((np.abs(flat) < 1).mean()),
    "mantel_r": obs_mantel, "mantel_p": pval,
    "anchor_coverage_percentile": cover_pct,
    "alpha_opt_mrrmse": a_mrr, "alpha_opt_mse": a_mse, "alpha_opt_r2": a_r2,
    "sse_share": shares, "row_sse_share": row_shares,
    "metric_geometry": sim.to_dict(orient="records"),
    "variance_components": vd.to_dict(orient="records"),
    "rank90": {tag: int(np.searchsorted(ev, 0.90) + 1) for ev, tag in curves},
}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("saved to", OUT)
