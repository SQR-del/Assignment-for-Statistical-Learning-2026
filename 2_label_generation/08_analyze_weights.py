"""08_analyze_weights.py  汇总 07_weights_experiment.R：自由度以外的差别是否全部来自 voom 权重，权重又为什么不同。
用法: python3 08_analyze_weights.py      输出: out/weights_summary.txt, out/weights_rows.csv
"""
import os, glob
import numpy as np, pandas as pd
from q2lib import *

OUT = Q2 + "/out"; lines = []
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)

rows = []
for ct in TNK:
    full = load_fit(Q2 + "/sim/" + tag(ct) + "/full"); dff = float(full["scal"].df_total); d = Q2 + "/weights_exp/" + tag(ct)
    for sd in sorted(glob.glob(d + "/r??_std")):
        r = int(os.path.basename(sd)[1:3])
        for var in ("std", "fullw", "fullwE"):
            f = load_fit(d + "/r%02d_%s" % (r, var))
            for s in f["compounds"]:
                i, j = f["row"][s], full["row"][s]; a, b = f["slp"][i], full["slp"][j]
                L = slp_from_t(f["t"][i], dff)                     # 只做自由度换算后的标签
                rows.append(dict(ct=ct, rep=r, variant=var, sm=s, rms_full=rms(b), ratio_raw=rms(b) / rms(a), ratio_after_df=rms(b) / rms(L),
                                 err_after_df=rms(L - b) / rms(b), su_ratio=float(np.median(f["su"][i] / full["su"][j])), t_ratio=rms(full["t"][j]) / rms(f["t"][i]),
                                 corr_t=float(np.corrcoef(f["t"][i], full["t"][j])[0, 1])))
R = pd.DataFrame(rows); R["strong"] = np.where(R.rms_full > 1.5, "strong", "weak"); R.to_csv(OUT + "/weights_rows.csv", index=False)

out("=" * 110); out("A. 干预实验：小模型换用大模型的权重后，与大模型还差多少")
out("   ratio_raw = 大模型行幅度 / 该拟合的行幅度；ratio_after_df = 先把该拟合的 t 值按大模型自由度换算成标签后的同一比值（=1 表示只差自由度）")
out("   err_after_df = 换算后与大模型标签的相对误差；su_ratio = 标准误系数之比（该拟合 / 大模型，逐基因中位数）")
A = R.groupby(["ct", "strong", "variant"]).agg(n=("sm", "size"), ratio_raw=("ratio_raw", "median"), ratio_after_df=("ratio_after_df", "median"),
      q10=("ratio_after_df", lambda x: x.quantile(.1)), q90=("ratio_after_df", lambda x: x.quantile(.9)), err_after_df=("err_after_df", "median"),
      su_ratio=("su_ratio", "median"), corr_t=("corr_t", "median"))
out(A.round(3).to_string()); A.to_csv(OUT + "/weights_tableA_intervention.csv")

out("\n" + "=" * 110); out("B. voom 曲线：同一表达水平下，小模型预期的噪声标准差 / 大模型预期的噪声标准差（各次重复的中位数，括号内为最小-最大）")
for ct in TNK:
    V = pd.read_csv(Q2 + "/weights_exp/" + tag(ct) + "/voom_lines.csv"); G1 = pd.read_csv(Q2 + "/weights_exp/" + tag(ct) + "/firstpass_group_sd.csv")
    qs = np.quantile(G1.sx, [.1, .25, .5, .75, .9]); sm = V[[c for c in V.columns if c.startswith("small_")]].to_numpy() / V[["full"]].to_numpy()
    s = "   %-20s" % ct
    for q, lab in zip(qs, ("10%", "25%", "50%", "75%", "90%")):
        k = int(np.argmin(np.abs(V.x - q))); s += "  基因分位 %s (x=%.1f): %.3f (%.3f-%.3f)" % (lab, q, np.median(sm[k]), sm[k].min(), sm[k].max())
    out(s)

out("\n" + "=" * 110); out("C. 大模型第一遍（不加权）拟合：各组样本的残差标准差之比（逐基因，按表达水平五等分后取中位数）")
out("   PC = 两种阳性对照；ORD = 普通药物。比值 < 1 表示该组比普通药物孔更安静")
for ct in TNK:
    G1 = pd.read_csv(Q2 + "/weights_exp/" + tag(ct) + "/firstpass_group_sd.csv"); G1 = G1[(G1.sd_ORD > 0) & (G1.sd_all > 0)]
    G1["bin"] = pd.qcut(G1.sx, 5, labels=["最低", "低", "中", "高", "最高"])
    t = G1.groupby("bin", observed=True).apply(lambda g: pd.Series(dict(sx=g.sx.median(), DMSO_vs_ORD=(g.sd_DMSO / g.sd_ORD).median(), PC_vs_ORD=(g.sd_PC / g.sd_ORD).median(),
        CTRL_vs_ORD=(np.sqrt((g.sd_DMSO ** 2 + 2 * g.sd_PC ** 2) / 3) / g.sd_ORD).median())))
    out("   " + ct); out(t.round(3).to_string())

out("\n" + "=" * 110); out("D. 大模型加权拟合：各组 实际残差方差 / voom 预测方差（2000 个随机基因的中位数；voom 的模型若对每组都成立，应都接近同一个值）")
for ct in TNK:
    D = pd.read_csv(Q2 + "/weights_exp/" + tag(ct) + "/weighted_group_ratio.csv"); D = D[D.ORD > 0]
    out("   %-20s DMSO %.3f | PC %.3f | ORD %.3f || DMSO/ORD %.3f  PC/ORD %.3f" % (ct, D.DMSO.median(), D.PC.median(), D.ORD.median(), (D.DMSO / D.ORD).median(), (D.PC / D.ORD).median()))

out("\n" + "=" * 110); out("E. 各组样本的规模（大模型）：样本数、每个样本的细胞数和总计数的中位数")
for ct in TNK:
    Sm = pd.read_csv(Q2 + "/weights_exp/" + tag(ct) + "/samples_full.csv")
    t = Sm.groupby("group").agg(n=("sample", "size"), n_cells=("n_cells", "median"), lib_size=("lib_size", "median"), mean_weight=("mean_weight", "median"))
    out("   " + ct); out(t.round(1).to_string())
open(OUT + "/weights_summary.txt", "w").write("\n".join(lines) + "\n")
