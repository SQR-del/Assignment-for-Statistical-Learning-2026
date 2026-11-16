"""中期报告图 2(b)：同一种药物在两种 T/NK 细胞之间响应的平均 Pearson 相关。
数据：corr_tnk.csv（按组员 eda/01_explore.py 的方法在服务器上计算：去掉阳性对照，140 种共有药物，逐药物求相关后平均）。
作图代码沿用组员 eda/01_explore.py，只把标题改短，避免被色条遮住。"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
labels = ["T cells CD4+", "T regulatory cells", "T cells CD8+", "NK cells"]
Cm = pd.read_csv("corr_tnk.csv", index_col=0).loc[labels, labels].to_numpy()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(Cm, vmin=0, vmax=0.3, cmap="viridis")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels(labels)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{Cm[i, j]:.2f}", ha="center", va="center",
                color="white" if Cm[i, j] < 0.18 else "black", fontsize=9)
ax.set_title("Cross-cell-type correlation")
fig.colorbar(im, ax=ax, label="mean Pearson r (shared drugs)")
fig.tight_layout(); fig.savefig("fig_corr.pdf"); fig.savefig("fig_corr.png", dpi=130)
