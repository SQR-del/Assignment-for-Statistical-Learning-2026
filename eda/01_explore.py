"""Exploratory data analysis for the single-cell perturbation labels.

Produces the figures 01..05 in this folder (label distribution, drug coverage,
per-cell-type response strength, cross-cell-type correlation, top genes).

Run:  python eda/01_explore.py
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import common as C  # noqa: E402

OUT = HERE
df = pd.read_parquet(C.data_path("de_train.parquet"))
genes = [c for c in df.columns if c not in ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]]
drug = df[~df.control].copy()
X = drug[genes].values
FULL = C.FULL_CELLS

# 1. label distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(np.abs(X).ravel(), bins=200, log=True)
ax[0].set_yscale("log"); ax[0].set_xlabel("|signed -log10(p)|"); ax[0].set_ylabel("count (log)")
ax[0].set_title("Distribution of label magnitude")
ax[1].hist(X.ravel(), bins=200)
ax[1].set_yscale("log"); ax[1].set_xlabel("signed -log10(p)"); ax[1].set_ylabel("count (log)")
ax[1].set_title("Signed label distribution")
fig.tight_layout(); fig.savefig(f"{OUT}/01_label_distribution.png", dpi=130); plt.close(fig)

# 2. coverage: drugs per cell type
cov = drug.groupby("cell_type")["sm_name"].nunique().sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
cov.plot.bar(ax=ax, color="steelblue")
ax.set_ylabel("# drugs observed"); ax.set_xlabel("cell type")
ax.set_title("Drug coverage per cell type (non-control)")
for i, v in enumerate(cov.values):
    ax.text(i, v + 1, str(v), ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/02_coverage.png", dpi=130); plt.close(fig)

# 3. per-cell-type response strength
ma = drug.assign(ma=np.abs(X).mean(1)).groupby("cell_type")["ma"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([ma.get_group(c).values for c in FULL + ["B cells", "Myeloid cells"]],
           tick_labels=FULL + ["B cells", "Myeloid cells"])
ax.set_ylabel("row mean |value|"); ax.set_title("Per-drug response strength by cell type")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout(); fig.savefig(f"{OUT}/03_strength.png", dpi=130); plt.close(fig)

# 4. cross-cell-type per-drug correlation heatmap
mats = {c: drug[drug.cell_type == c].set_index("sm_name")[genes] for c in FULL}
shared = sorted(set.intersection(*[set(m.index) for m in mats.values()]))


def rowcorr(a, b):
    return np.nanmean([np.corrcoef(mats[a].loc[d].values, mats[b].loc[d].values)[0, 1] for d in shared])


labels = FULL
Cm = np.full((4, 4), np.nan)
for i, j in itertools.product(range(4), repeat=2):
    Cm[i, j] = 1.0 if i == j else rowcorr(labels[i], labels[j])
    Cm[j, i] = Cm[i, j]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(Cm, vmin=0, vmax=0.3, cmap="viridis")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels(labels)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{Cm[i, j]:.2f}", ha="center", va="center",
                color="white" if Cm[i, j] < 0.18 else "black", fontsize=9)
ax.set_title("Per-drug response correlation across cell types")
fig.colorbar(im, ax=ax, label="mean Pearson r (shared drugs)")
fig.tight_layout(); fig.savefig(f"{OUT}/04_crosscell_corr.png", dpi=130); plt.close(fig)

# 5. top variable genes
gabs = pd.Series(np.abs(X).mean(0), index=genes)
fig, ax = plt.subplots(figsize=(8, 5))
gabs.sort_values(ascending=False).head(25)[::-1].plot.barh(ax=ax, color="indianred")
ax.set_xlabel(f"mean |value| across {len(drug)} drug rows"); ax.set_ylabel("gene")
ax.set_title("Top 25 most 'responsive' genes overall")
fig.tight_layout(); fig.savefig(f"{OUT}/05_top_genes.png", dpi=130); plt.close(fig)

print("saved figures to", OUT)
print("shared drugs among 4 full cell types:", len(shared))
print("top genes:", list(gabs.sort_values(ascending=False).head(15).index))
