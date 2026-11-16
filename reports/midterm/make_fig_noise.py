"""中期报告图 5：按样本分组的噪声差异，以及由它推算的标签之比。
数据：../../2_label_generation/out/group_variance_summary.txt（09_group_variance.py 的输出；各组残差方差 / voom 预测方差，
3000 个基因的中位数，区间为对基因重抽样的 95% 区间）。风格与组员 eda/01_explore.py 一致：matplotlib 默认样式。"""
import re, io
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
txt = io.open("../../2_label_generation/out/group_variance_summary.txt", encoding="utf-8").read()
order = ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells", "B cells", "Myeloid cells"]
num = r"([\d.]+)"
D = {}
for c in order:
    line = next(l for l in txt.splitlines() if l.startswith(c))
    g = lambda pat: [float(x) for x in re.search(pat, line).groups()]
    D[c] = dict(dmso=g(r"DMSO/ORD %s \(%s-%s\)" % (num, num, num)), pc=g(r"PC/ORD %s \(%s-%s\)" % (num, num, num)),
                pred=g(r"预测标签 大/小 %s \(%s-%s\)" % (num, num, num)),
                sim=(g(r"模拟实测（弱行）%s" % num)[0] if "模拟实测" in line else np.nan))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(len(order)); w = 0.38
a = ax[0]
for k, (key, lab, col) in enumerate((("dmso", "DMSO wells / drug wells", "C0"), ("pc", "positive-control wells / drug wells", "C1"))):
    v = np.array([D[c][key] for c in order])
    a.bar(x + (k - 0.5) * w, v[:, 0] - 1, w, bottom=1, color=col, label=lab, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=3)
a.axhline(1, color="black", lw=0.8); a.set_ylim(0.7, 1.72)
a.set_xticks(x); a.set_xticklabels(order, rotation=30, ha="right")
a.set_ylabel("ratio of noise variance"); a.set_title("(a) Noise of control wells relative to drug wells"); a.legend(loc="upper left")

b = ax[1]
pred = np.array([D[c]["pred"] for c in order]); sim = np.array([D[c]["sim"] for c in order])
b.bar(x - w / 2, pred[:, 0] - 1, w, bottom=1, color="C0", label="predicted from group noise", yerr=[pred[:, 0] - pred[:, 1], pred[:, 2] - pred[:, 0]], capsize=3)
b.bar(x + w / 2, sim - 1, w, bottom=1, color="C1", label="simulation (observed)")
for i in np.flatnonzero(np.isnan(sim)):
    b.text(x[i] + w / 2, 1.004, "n/a", ha="center", va="bottom", fontsize=8, color="gray")
for xs, vals, ends in ((x - w / 2, pred[:, 0], pred[:, 1:]), (x + w / 2, sim, np.stack([sim, sim], 1))):
    for xi, v, (lo, hi) in zip(xs, vals, ends):
        if np.isfinite(v):
            up = v >= 1
            b.text(xi, (hi + 0.004) if up else (lo - 0.004), "%.2f" % v, ha="center", va="bottom" if up else "top", fontsize=8)
b.axhline(1, color="black", lw=0.8); b.set_ylim(0.85, 1.23); b.set_xlim(-0.6, len(order) - 0.4); a.set_xlim(-0.6, len(order) - 0.4)
b.set_xticks(x); b.set_xticklabels(order, rotation=30, ha="right")
b.set_ylabel("full-fit label / small-fit label"); b.set_title("(b) Label ratio on weak rows"); b.legend(loc="upper left")
fig.tight_layout(); fig.savefig("fig_noise.pdf"); fig.savefig("fig_noise.png", dpi=130)
for c in order: print(c, D[c])
