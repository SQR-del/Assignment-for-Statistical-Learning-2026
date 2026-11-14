"""Figures for the Limma reproduction, calibration and error decomposition."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import CELLS, FIG, OUT, SEED, SHORT, ensure_out, load_de, load_genes

ensure_out()
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9})
GENES = load_genes()
DE = load_de()
rng = np.random.default_rng(SEED)


def repro(cell):
    f = os.path.join(OUT, "repro_" + cell.replace(" ", "_").replace("+", "p") + ".npz")
    return np.load(f, allow_pickle=True)


fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
xs, ys = [], []
for cell in CELLS:
    z = repro(cell)
    d = DE[DE.cell_type == cell].set_index("sm_name").loc[z["drugs"], GENES].to_numpy(float)
    xs.append(z["full"].ravel())
    ys.append(d.ravel())
x = np.concatenate(xs)
y = np.concatenate(ys)
ok = np.isfinite(x) & np.isfinite(y)
x, y = x[ok], y[ok]
idx = rng.choice(len(x), size=min(400000, len(x)), replace=False)
hb = ax[0].hexbin(x[idx], y[idx], gridsize=60, bins="log", cmap="viridis", mincnt=1)
lim = [np.quantile(x, 0.001), np.quantile(x, 0.999)]
ax[0].plot(lim, lim, "r--", lw=1)
ax[0].set_xlim(lim)
ax[0].set_ylim(lim)
ax[0].set_xlabel("reproduced signed -log10 p (pseudobulk Limma)")
ax[0].set_ylabel("official de_train label")
ax[0].set_title("Reproduction of Limma labels (pooled genes/drugs)")
fig.colorbar(hb, ax=ax[0], fraction=0.04, label="log10 count")
rep = pd.read_csv(os.path.join(OUT, "repro_by_row.csv"))
m = rep.groupby("cell_type")[["pearson", "spearman"]].mean().reindex(CELLS)
xp = np.arange(len(CELLS))
ax[1].bar(xp - 0.2, m.pearson, width=0.4, label="Pearson", color="#4C72B0")
ax[1].bar(xp + 0.2, m.spearman, width=0.4, label="Spearman", color="#DD8452")
ax[1].set_xticks(xp)
ax[1].set_xticklabels([SHORT[c] for c in CELLS], rotation=0)
ax[1].set_ylim(0, 1)
ax[1].set_ylabel("mean per-row correlation vs de_train")
ax[1].set_title("Reproduction quality by cell type")
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_repro.png"))
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
xs, ys = [], []
for cell in CELLS:
    z = repro(cell)
    xs.append(z["small"].ravel())
    ys.append(z["full"].ravel())
x = np.concatenate(xs)
y = np.concatenate(ys)
ok = np.isfinite(x) & np.isfinite(y)
x, y = x[ok], y[ok]
idx = rng.choice(len(x), size=min(400000, len(x)), replace=False)
hb = ax[0].hexbin(x[idx], y[idx], gridsize=60, bins="log", cmap="magma", mincnt=1)
lim = [np.quantile(x, 0.001), np.quantile(x, 0.999)]
ax[0].plot(lim, lim, "c--", lw=1)
ax[0].set_xlim(lim)
ax[0].set_ylim(lim)
ax[0].set_xlabel("small model: half cells")
ax[0].set_ylabel("large model: all cells")
ax[0].set_title("Small vs large Limma model (pooled)")
fig.colorbar(hb, ax=ax[0], fraction=0.04, label="log10 count")
hf = pd.read_csv(os.path.join(OUT, "half_vs_full.csv"))
m = hf.groupby("cell_type")[["corr_small_large", "slope_large_on_small",
                              "amp_ratio"]].mean().reindex(CELLS)
xp = np.arange(len(CELLS))
ax[1].bar(xp - 0.2, m.corr_small_large, width=0.4, label="corr", color="#55A868")
ax[1].bar(xp + 0.2, m.slope_large_on_small, width=0.4,
          label="slope large~small", color="#C44E52")
ax[1].plot(xp, m.amp_ratio, "ko", ms=4, label="amplitude ratio")
ax[1].axhline(1.0, color="k", lw=0.6, ls=":")
ax[1].set_xticks(xp)
ax[1].set_xticklabels([SHORT[c] for c in CELLS])
ax[1].set_ylim(0, 1.15)
ax[1].set_title("Small vs large: agreement and attenuation")
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_small_large.png"))
plt.close(fig)

cal = pd.read_csv(os.path.join(OUT, "calibration_results.csv"))
m = cal.groupby("model")[["base_mrrmse", "mrrmse_cv", "mrrmse_oracle"]].mean()
m = m.sort_values("base_mrrmse")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
xp = np.arange(len(m))
ax[0].bar(xp - 0.2, m.base_mrrmse, width=0.4, label="no calibration", color="0.7")
ax[0].bar(xp + 0.2, m.mrrmse_cv, width=0.4, label="CV amplitude calibration", color="#4C72B0")
ax[0].plot(xp, m.mrrmse_oracle, "r.", ms=6, label="oracle")
ax[0].set_xticks(xp)
ax[0].set_xticklabels(m.index, rotation=35, ha="right", fontsize=7)
ax[0].set_ylabel("mrrmse")
ax[0].set_title("Amplitude calibration (P1, 4 folds)")
ax[0].legend(frameon=False, fontsize=7)
a = cal.groupby("model")[["alpha_cv", "alpha_oracle"]].mean().reindex(m.index)
ax[1].barh(np.arange(len(a)), a.alpha_cv, color="#4C72B0", label="CV alpha")
ax[1].plot(a.alpha_oracle, np.arange(len(a)), "r.", label="oracle alpha")
ax[1].axvline(0.81, color="k", ls="--", lw=1, label="small/large slope 0.81")
ax[1].axvline(0.955, color="k", ls=":", lw=1, label="amplitude 0.955")
ax[1].set_yticks(np.arange(len(a)))
ax[1].set_yticklabels(a.index, fontsize=7)
ax[1].invert_yaxis()
ax[1].set_xlabel("optimal scaling alpha")
ax[1].set_title("Optimal amplitude factor is strong shrinkage")
ax[1].legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_calibration.png"))
plt.close(fig)

noise = pd.read_csv(os.path.join(OUT, "noise_by_row.csv"))
dec = pd.read_csv(os.path.join(OUT, "error_decomposition.csv"))
dec = dec[dec.noise_kind == "technical"]
model_mrr = dec.groupby("cell_type").mrrmse.mean()
lab_rms = {}
for cell in CELLS:
    d = DE[DE.cell_type == cell][GENES].to_numpy(float)
    lab_rms[cell] = np.sqrt(np.mean(d ** 2))
noise = noise.groupby("cell_type").mean(numeric_only=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
xp = np.arange(len(CELLS))
ax[0].bar(xp - 0.25, [lab_rms[c] for c in CELLS], width=0.25,
          label="label RMS", color="0.75")
ax[0].bar(xp, [np.sqrt(noise.loc[c, "noise_technical"]) for c in CELLS],
          width=0.25, label="technical noise floor", color="#C44E52")
ax[0].bar(xp + 0.25, [np.sqrt(noise.loc[c, "noise_donor"]) for c in CELLS],
          width=0.25, label="donor noise floor", color="#DD8452")
ax[0].set_xticks(xp)
ax[0].set_xticklabels([SHORT[c] for c in CELLS])
ax[0].set_ylabel("RMSE scale")
ax[0].set_title("Label RMS vs estimated noise floors")
ax[0].legend(frameon=False, fontsize=7)
frac = [noise.loc[c, "noise_technical"] / lab_rms[c] ** 2 for c in CELLS]
ax[1].bar(xp, frac, color="#55A868")
ax[1].set_xticks(xp)
ax[1].set_xticklabels([SHORT[c] for c in CELLS])
ax[1].set_ylabel("noise variance / label variance")
ax[1].set_title("Estimated noise fraction")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_noise.png"))
plt.close(fig)

m = dec.groupby("model")[["magnitude_share", "noise_share",
                          "direction_share"]].mean()
m = m.sort_values("direction_share")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
xp = np.arange(len(m))
ax[0].barh(xp, m.magnitude_share, color="#DD8452", label="magnitude")
ax[0].barh(xp, m.noise_share, left=m.magnitude_share, color="#55A868", label="noise")
ax[0].barh(xp, m.direction_share,
           left=m.magnitude_share + m.noise_share, color="#4C72B0", label="direction")
ax[0].set_yticks(xp)
ax[0].set_yticklabels(m.index, fontsize=7)
ax[0].set_xlim(0, 1)
ax[0].set_xlabel("share of mean squared error")
ax[0].set_title("Error decomposition by model")
ax[0].legend(frameon=False, fontsize=7, loc="lower right")
cells4 = ["T cells CD4+", "T regulatory cells", "T cells CD8+", "NK cells"]
c = dec.groupby("cell_type")[["magnitude_share", "noise_share",
                              "direction_share"]].mean().reindex(cells4)
xp = np.arange(len(c))
ax[1].bar(xp, c.magnitude_share, color="#DD8452", label="magnitude")
ax[1].bar(xp, c.noise_share, bottom=c.magnitude_share, color="#55A868", label="noise")
ax[1].bar(xp, c.direction_share,
          bottom=c.magnitude_share + c.noise_share, color="#4C72B0", label="direction")
ax[1].set_xticks(xp)
ax[1].set_xticklabels([SHORT[x] for x in cells4])
ax[1].set_ylim(0, 1)
ax[1].set_title("Error decomposition by held-out cell")
ax[1].legend(frameon=False, fontsize=7, loc="upper center",
             bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_decomposition.png"))
plt.close(fig)
print("saved figures to", FIG)
