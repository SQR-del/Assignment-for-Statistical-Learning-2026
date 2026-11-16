"""中期报告的两张图（风格与组员 eda/01_explore.py 一致：matplotlib 默认样式）。
fig_rowrms：各行标签的均方根（第 1 节），数据 row_rms.csv（来自 de_train）；
fig_ratio：大模型 / 小模型标签之比（第 2 节），数据 ../../2_label_generation/out/sim_gene_bins.csv。"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv("row_rms.csv"); R = R[~R.control]
G = pd.read_csv("../../2_label_generation/out/sim_gene_bins.csv")

fig, a = plt.subplots(figsize=(7, 4.2))
order = ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells", "B cells", "Myeloid cells"]
rng = np.random.default_rng(0)
for i, c in enumerate(order):
    y = R[R.cell_type == c].rms.to_numpy()
    a.scatter(i + rng.uniform(-0.25, 0.25, len(y)), y, s=8, color="C0", alpha=0.5)
    a.plot([i - 0.32, i + 0.32], [np.median(y)] * 2, color="black", lw=2)
a.axhline(np.sqrt(2) / np.log(10), color="gray", ls="--", lw=1, label="no effect (0.61)")
a.set_yscale("log"); a.set_yticks([0.5, 1, 2, 5, 10]); a.set_yticklabels(["0.5", "1", "2", "5", "10"])
a.set_xticks(range(6)); a.set_xticklabels(order, rotation=30, ha="right")
a.set_ylabel("row RMS of labels"); a.set_title("Response strength per row (bar = median)"); a.legend(loc="upper right")
fig.tight_layout(); fig.savefig("fig_rowrms.pdf"); fig.savefig("fig_rowrms.png", dpi=130); plt.close(fig)

fig, b = plt.subplots(figsize=(7, 4.2))
bins = ["0-1", "1-2", "2-3", "3-5", "5-8", "8-12", "12-20", "20-40"]; x = np.arange(len(bins))
for c, col in (("NK cells", "C0"), ("T cells CD4+", "C1")):
    g = G[G.ct == c].set_index("label_small").loc[bins]
    b.fill_between(x, g.ci_lo, g.ci_hi, color=col, alpha=0.2, lw=0)
    b.plot(x, g.observed, color=col, marker="o", label=c + " (observed)")
g = G[G.ct == "NK cells"].set_index("label_small").loc[bins]
b.plot(x, g.df_only, color="gray", ls="--", label="degrees-of-freedom prediction")
b.axhline(1, color="black", lw=0.6); b.set_xticks(x); b.set_xticklabels(bins)
b.set_xlabel("|label| in the small fit"); b.set_ylabel("full-fit label / small-fit label")
b.set_title("Label ratio: full fit vs. small fit"); b.legend(loc="lower left")
fig.tight_layout(); fig.savefig("fig_ratio.pdf"); fig.savefig("fig_ratio.png", dpi=130)
