"""10_noise_sources.py  检验"对照孔与普通药物孔噪声不同"的四个可能原因。
用法: Rscript 10_noise_sources.R 6 && python3 10_noise_sources.py      输出: out/noise_sources_summary.txt
q = 每个样本的平均（权重 x 残差平方 / (1-杠杆值)），可理解为"这个样本相对模型预期有多吵"。
"""
import numpy as np, pandas as pd
from q2lib import *
R = pd.read_csv(Q2 + "/out/noise_samples.csv"); lines = []
# 以"对基因取中位数"的 q 为准：对基因取均值会被少数极端基因（如 B 细胞的免疫球蛋白基因）主导
for m in ("base", "col", "pcdonor"): R["qmean_" + m] = R["q_" + m]; R["q_" + m] = R["qmed_" + m]
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
rng = np.random.default_rng(0)
def ratio_ci(a, b, B=2000):
    """两组样本 q 的均值之比及自助法 95% 区间"""
    a, b = np.asarray(a), np.asarray(b); v = [rng.choice(a, len(a)).mean() / rng.choice(b, len(b)).mean() for _ in range(B)]
    return a.mean() / b.mean(), np.quantile(v, .025), np.quantile(v, .975)
fmt = lambda t: "%.2f (%.2f-%.2f)" % t

out("=" * 110); out("0. 板的布局：对照孔在哪些列")
g = R[R.ct == "NK cells"]
out("   DMSO 列:", sorted(g[g.group == "DMSO"].col.unique()), "| 阳性对照 列:", {p: sorted(g[g.sm_name == p].col.unique()) for p in PC}, "| 普通药 列:", sorted(g[g.group == "ORD"].col.unique()))

out("\n" + "=" * 110); out("1. 三种均值模型下，各组相对普通药物孔有多吵（样本 q 的均值之比，括号内为对样本重抽样的 95% 区间）")
out("   base = 官方模型；col = 加上列；pcdonor = 加上 阳性对照 x 供体")
for ct in ALL_CT:
    g = R[R.ct == ct]; o = g[g.group == "ORD"]
    for m in ("base", "col", "pcdonor"):
        out("   %-20s %-8s DMSO/普通 %s | Belinostat/普通 %s | Dabrafenib/普通 %s" % (ct, m, fmt(ratio_ci(g[g.group == "DMSO"]["q_" + m], o["q_" + m])),
            fmt(ratio_ci(g[g.sm_name == "Belinostat"]["q_" + m], o["q_" + m])), fmt(ratio_ci(g[g.sm_name == "Dabrafenib"]["q_" + m], o["q_" + m]))))

out("\n" + "=" * 110); out("2. 列位置：官方模型下，普通药物孔按列的平均 q（相对该细胞类型普通孔的总体均值）；最后给出加上列之后普通孔 q 的下降比例")
for ct in ALL_CT:
    g = R[(R.ct == ct) & (R.group == "ORD")]
    c = (g.groupby("col").q_base.mean() / g.q_base.mean()).round(2)
    out("   %-20s %s | 加列后普通孔 q: %.3f -> %.3f，对照孔 q: %.3f -> %.3f" % (ct, " ".join("%d:%.2f" % (k, v) for k, v in c.items()), g.q_base.mean(), g.q_col.mean(),
        R[(R.ct == ct) & (R.group != "ORD")].q_base.mean(), R[(R.ct == ct) & (R.group != "ORD")].q_col.mean()))

out("\n" + "=" * 110); out("3. 药物强弱：普通药按官方标签的行幅度分成弱 / 中 / 强，每组的平均 q 相对 DMSO 孔（官方模型）")
de, Y, key = official(); strength = {k: rms(Y[i]) for k, i in key.items()}
R["strength"] = [strength.get((c, s), np.nan) for c, s in zip(R.ct, R.sm_name)]
for ct in ALL_CT:
    g = R[R.ct == ct]; o = g[g.group == "ORD"].copy(); d = g[g.group == "DMSO"].q_base
    o["bin"] = pd.cut(o.strength, [0, 0.8, 1.5, np.inf], labels=["弱(<0.8)", "中(0.8-1.5)", "强(>1.5)"])
    s = "   %-20s" % ct
    for b, x in o.groupby("bin", observed=True): s += "  %s: n=%d  %s" % (b, len(x), fmt(ratio_ci(x.q_base, d)))
    out(s + "  | 普通药 q 与行幅度的秩相关 %.2f" % o[["q_base", "strength"]].corr(method="spearman").iloc[0, 1])

out("\n" + "=" * 110); out("4. 细胞数：各组每个样本的细胞数（中位数），以及在各组内部 q 与细胞数的秩相关（官方模型）")
for ct in ALL_CT:
    g = R[R.ct == ct]; s = "   %-20s" % ct
    for k, nm in (("DMSO", "DMSO"), ("Belinostat", "Belinostat"), ("Dabrafenib", "Dabrafenib"), ("ORD", "普通药")):
        x = g[g.group == k] if k in ("DMSO", "ORD") else g[g.sm_name == k]
        s += "  %s: %d 孔, 细胞 %.0f, 相关 %.2f" % (nm, len(x), x.n_cells.median(), x[["q_base", "n_cells"]].corr(method="spearman").iloc[0, 1])
    out(s)
out("   按细胞数分档的平均 q（全部样本合并，相对该细胞类型 DMSO 孔均值）")
for ct in ALL_CT:
    g = R[R.ct == ct].copy(); g["b"] = pd.cut(g.n_cells, [0, 5, 10, 20, 50, 100, 1e9], labels=["1-5", "6-10", "11-20", "21-50", "51-100", ">100"])
    t = g.groupby("b", observed=True).agg(n=("q_base", "size"), q=("q_base", "mean")); t["q"] /= g[g.group == "DMSO"].q_base.mean()
    out("   %-20s %s" % (ct, "  ".join("%s: n=%d q=%.2f" % (k, r.n, r.q) for k, r in t.iterrows())))

out("\n" + "=" * 110); out("5. 细胞类型比例：每个孔里 CD8 / Treg / CD4 占全部 T 细胞的比例，各组的均值与孔间标准差（先减去各供体的 DMSO 均值）")
S = pd.read_csv(Q2 + "/cache_official/samples.csv"); S["wellid"] = S.plate_name + "_" + S.well
T = S[S.cell_type.isin(["T cells CD4+", "T cells CD8+", "T regulatory cells"])].pivot_table(index="wellid", columns="cell_type", values="n_cells", fill_value=0)
info = S.drop_duplicates("wellid").set_index("wellid")[["sm_name", "donor_id"]]; T = T.join(info); T["tot"] = T.iloc[:, :3].sum(1)
T["grp"] = np.where(T.sm_name == DMSO, "DMSO", np.where(T.sm_name.isin(PC), T.sm_name, "普通药"))
for c in ["T cells CD4+", "T cells CD8+", "T regulatory cells"]:
    T["f"] = T[c] / T.tot; base = T[T.grp == "DMSO"].groupby("donor_id").f.mean(); T["dev"] = T.f - T.donor_id.map(base)
    out("   %-20s " % c + "  ".join("%s: 比例 %.3f, 孔间标准差 %.3f, T 细胞总数 %.0f" % (k, x.f.mean(), x.dev.std(), x.tot.median()) for k, x in T.groupby("grp")))
out("   5b. Belinostat 孔按供体：有 CD8 细胞的孔数 / 总孔数，CD8 占 T 细胞的平均比例（DMSO 孔的比例作对照）")
Tb = T.copy(); Tb["f8"] = Tb["T cells CD8+"] / Tb.tot; Tb["fr"] = Tb["T regulatory cells"] / Tb.tot
for dnr, x in Tb.groupby("donor_id"):
    b = x[x.grp == "Belinostat"]; d = x[x.grp == "DMSO"]
    out("   %s  Belinostat: %d/%d 孔有 CD8, CD8 比例 %.3f, Treg 比例 %.3f | DMSO: %d/%d 孔有 CD8, CD8 比例 %.3f, Treg 比例 %.3f" % (dnr, (b["T cells CD8+"] > 0).sum(), len(b), b.f8.mean(), b.fr.mean(), (d["T cells CD8+"] > 0).sum(), len(d), d.f8.mean(), d.fr.mean()))
open(Q2 + "/out/noise_sources_summary.txt", "w").write("\n".join(lines) + "\n")

# ---------------- 补充 ----------------
out("\n" + "=" * 110); out("6. 补充")
out("   模型的秩（参数个数）: " + "; ".join("%s base %d / col %d / pcdonor %d" % (ct, g.rank_base.iloc[0], g.rank_col.iloc[0], g.rank_pcdonor.iloc[0]) for ct, g in R.groupby("ct", sort=False)))
out("   每种普通药出现在几个不同的列（NK）: " + str(R[(R.ct == "NK cells") & (R.group == "ORD")].groupby("sm_name").col.nunique().value_counts().to_dict()))
out("   6a. 只看细胞数 > 100 的孔：强药（行幅度 > 1.5）相对 DMSO")
for ct in ("NK cells", "T cells CD4+"):
    g = R[(R.ct == ct) & (R.n_cells > 100)]; x = g[(g.group == "ORD") & (g.strength > 1.5)]
    out("   %-20s 强药 n=%d  %s" % (ct, len(x), fmt(ratio_ci(x.q_base, g[g.group == "DMSO"].q_base))))
out("   6b. 普通药里 q 最大的 6 种（各药 3 个孔的平均 q / DMSO 均值；括号内为细胞数中位数和官方标签行幅度）")
for ct in TNK:
    g = R[R.ct == ct]; d = g[g.group == "DMSO"].q_base.mean()
    t = g[g.group == "ORD"].groupby("sm_name").agg(q=("q_base", "mean"), cells=("n_cells", "median"), st=("strength", "first")).sort_values("q", ascending=False).head(6)
    out("   %-20s %s" % (ct, "; ".join("%s %.1f (细胞 %d, 幅度 %.1f)" % (k[:18], r.q / d, r.cells, r.st) for k, r in t.iterrows())))
out("   6c. 去掉这些极端药之后：普通药中 q 不在前 5% 的孔，相对 DMSO")
for ct in ALL_CT:
    g = R[R.ct == ct]; o = g[g.group == "ORD"]; x = o[o.q_base <= o.q_base.quantile(.95)]
    out("   %-20s %s | 前 5%% 的孔占普通孔 q 总和的 %.0f%%" % (ct, fmt(ratio_ci(x.q_base, g[g.group == "DMSO"].q_base)), 100 * o[o.q_base > o.q_base.quantile(.95)].q_base.sum() / o.q_base.sum()))
open(Q2 + "/out/noise_sources_summary.txt", "w").write("\n".join(lines) + "\n")
