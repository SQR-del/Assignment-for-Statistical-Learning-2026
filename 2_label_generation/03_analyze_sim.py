"""03_analyze_sim.py  汇总随机抽药模拟：行级放大倍数、逐基因曲线、机制拆分、区间。
用法: python3 03_analyze_sim.py
输出: out/sim_rows.csv  out/sim_gene_bins.csv  out/sim_summary.txt
只统计普通药物（每种药 3 个孔），阳性对照不计入。
"""
import os, sys, glob
import numpy as np, pandas as pd
from q2lib import *

OUT = Q2 + "/out"; os.makedirs(OUT, exist_ok=True)
BINS = [0, 1, 2, 3, 5, 8, 12, 20, 40, np.inf]
STR_BINS = [0, 0.8, 1.5, 3, np.inf]; STR_LAB = ["<0.8", "0.8-1.5", "1.5-3", ">3"]
rng = np.random.default_rng(0)
rows, gb = [], []

for ct in TNK:
    d = Q2 + "/sim/" + tag(ct)
    full = load_fit(d + "/full"); full_np = load_fit(d + "/full_noPC")
    reps = sorted(int(os.path.basename(x)[-2:]) for x in glob.glob(d + "/small_r??"))
    for r in reps:
        for var, ref in (("small", full), ("small_noPC", full_np), ("mid", full)):
            f = load_fit(d + "/%s_r%02d" % (var, r)); dfs, dff = float(f["scal"].df_total), float(ref["scal"].df_total)
            sig = np.sqrt(f["s2post"] / ref["s2post"])                      # 逐基因的噪声标准差之比（小/大）
            for s in f["compounds"]:
                if s in PC: continue
                i, j = f["row"][s], ref["row"][s]
                a, b, ta, tb = f["slp"][i], ref["slp"][j], f["t"][i], ref["t"][j]
                # 逐步把小模型的量换成大模型的量；只在噪声比正常的基因上做（全零等退化基因的方差比不稳定，排除）
                ok = (sig > 1 / 3) & (sig < 3) & np.isfinite(sig)
                L0 = slp_from_t(ta, dff)                                    # 只换自由度
                L1 = slp_from_t(ta * np.where(ok, sig, 1.0), dff)           # 再换噪声标准差
                L2 = slp_from_t(ta * np.where(ok, sig * f["su"][i] / ref["su"][j], 1.0), dff)   # 再换设计因子；剩下的差别来自效应估计
                rows.append(dict(ct=ct, rep=r, variant=var, sm=s, wells=f["wells"][i], rms_small=rms(a), rms_full=rms(b), ratio=rms(b) / rms(a),
                                 part_df=rms(L0) / rms(a), part_sigma=rms(L1) / rms(L0), part_su=rms(L2) / rms(L1), part_coef=rms(b) / rms(L2),
                                 t_ratio=rms(tb) / rms(ta), sigma_ratio_med=float(np.median(sig)), su_ratio_med=float(np.median(f["su"][i] / ref["su"][j])), coef_ratio_strong=float(np.median((ref["coef"][j] / f["coef"][i])[np.abs(a) > 3])) if (np.abs(a) > 3).sum() > 20 else np.nan, share_ok=float(ok.mean()),
                                 corr=float(np.corrcoef(a, b)[0, 1]), err_before=rms(a - b) / rms(b), err_df=rms(L0 - b) / rms(b), err_df_sigma=rms(L1 - b) / rms(b),
                                 df_small=dfs, df_full=dff))
                k = np.digitize(np.abs(a), BINS) - 1; sg = np.sign(a)
                for q in range(len(BINS) - 1):
                    m = k == q
                    if m.any(): gb.append((ct, r, var, s, q, int(m.sum()), float(np.abs(a[m]).sum()), float((b[m] * sg[m]).sum()), float((L0[m] * sg[m]).sum()), float((L1[m] * sg[m]).sum())))
    print(ct, "done, reps:", len(reps), flush=True)

R = pd.DataFrame(rows); R["strength"] = pd.cut(R.rms_small, STR_BINS, labels=STR_LAB)
G = pd.DataFrame(gb, columns=["ct", "rep", "variant", "sm", "bin", "n", "sum_small", "sum_full", "sum_df", "sum_df_sigma"])
R.to_csv(OUT + "/sim_rows.csv", index=False)


def cboot(df, fun, B=1000):
    """按药物整体重抽样（同一种药在多次重复中出现，不能当成独立行）"""
    grp = {s: g for s, g in df.groupby("sm")}; names = list(grp)
    if len(names) < 3: return (np.nan, np.nan)
    v = [fun(pd.concat([grp[s] for s in rng.choice(names, len(names))])) for _ in range(B)]
    return tuple(np.nanquantile(v, [0.025, 0.975]))


lines = []
def out(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)

S = R[R.variant == "small"]
out("=" * 100); out("表 1  行级放大倍数（大模型行均方根 / 小模型行均方根），普通药物，small 对 full")
out("      区间 = 按药物整体重抽样的 95% 区间；n_drugs = 不同药物数")
t1 = []
for ct in TNK + ["all"]:
    for lab in STR_LAB:
        g = S[(S.strength == lab) & ((S.ct == ct) | (ct == "all"))]
        if len(g) == 0: continue
        lo, hi = cboot(g, lambda x: x.ratio.median())
        t1.append(dict(ct=ct, rms_small=lab, n_rows=len(g), n_drugs=g.sm.nunique(), median=g.ratio.median(), q25=g.ratio.quantile(.25), q75=g.ratio.quantile(.75), ci_lo=lo, ci_hi=hi, share_gt_1p1=(g.ratio > 1.1).mean()))
T1 = pd.DataFrame(t1); out(T1.round(3).to_string(index=False)); T1.to_csv(OUT + "/sim_table1_row_ratio.csv", index=False)

out("\n" + "=" * 100); out("表 2  放大倍数的来源拆分（中位数）：总倍数 = 自由度 x 噪声标准差 x 设计因子 x 效应估计")
out("      sigma_ratio / su_ratio = 小模型与大模型的噪声标准差之比 / 设计因子之比（逐基因中位数）；coef_ratio = 显著基因上 大/小 效应之比的中位数")
t2 = S.groupby(["ct", "strength"], observed=True).agg(n=("ratio", "size"), total=("ratio", "median"), df=("part_df", "median"), sigma=("part_sigma", "median"),
      design=("part_su", "median"), effect=("part_coef", "median"), t_ratio=("t_ratio", "median"), sigma_ratio=("sigma_ratio_med", "median"), su_ratio=("su_ratio_med", "median"), coef_ratio=("coef_ratio_strong", "median"))
out(t2.round(3).to_string()); t2.to_csv(OUT + "/sim_table2_decomposition.csv")

out("\n" + "=" * 100); out("表 3  三种小模型对比（中位数）。small = 15 药 + 阳性对照；small_noPC = 15 药，无阳性对照（参照为去掉阳性对照的大模型）；mid = 60 药 + 阳性对照")
t3 = R.groupby(["ct", "variant", "strength"], observed=True).agg(n=("ratio", "size"), df_small=("df_small", "median"), df_full=("df_full", "median"), total=("ratio", "median"),
      df=("part_df", "median"), sigma=("part_sigma", "median"), effect=("part_coef", "median"), sigma_ratio=("sigma_ratio_med", "median"))
t3 = t3[t3.index.get_level_values("strength").isin(["1.5-3", ">3", "<0.8"])]
out(t3.round(3).to_string()); t3.to_csv(OUT + "/sim_table3_variants.csv")

out("\n" + "=" * 100); out("表 4  与大模型标签的相对误差（中位数）：不校正 / 只换自由度 / 再换噪声标准差")
t4 = S.groupby(["ct", "strength"], observed=True).agg(n=("ratio", "size"), corr=("corr", "median"), err_before=("err_before", "median"), err_df=("err_df", "median"), err_df_sigma=("err_df_sigma", "median"))
out(t4.round(3).to_string()); t4.to_csv(OUT + "/sim_table4_error.csv")

out("\n" + "=" * 100); out("表 5  逐基因曲线：按小模型标签绝对值分档，大模型标签（带符号）均值 / 小模型标签绝对值均值")
out("      observed = 实际；df_only = 只换自由度的预测；df_sigma = 再换噪声标准差的预测；区间按药物重抽样")
blab = ["%g-%g" % (BINS[i], BINS[i + 1]) for i in range(len(BINS) - 1)]
t5 = []
Gs = G[G.variant == "small"]
for ct in TNK + ["all"]:
    g0 = Gs if ct == "all" else Gs[Gs.ct == ct]
    for q in range(len(BINS) - 1):
        g = g0[g0.bin == q]
        if g.n.sum() < 50: continue
        f = lambda x: x.sum_full.sum() / x.sum_small.sum()
        lo, hi = cboot(g, f, B=500)
        t5.append(dict(ct=ct, label_small=blab[q], n_entries=int(g.n.sum()), n_drugs=g.sm.nunique(), mean_small=g.sum_small.sum() / g.n.sum(), observed=f(g), ci_lo=lo, ci_hi=hi,
                       df_only=g.sum_df.sum() / g.sum_small.sum(), df_sigma=g.sum_df_sigma.sum() / g.sum_small.sum()))
T5 = pd.DataFrame(t5); out(T5.round(3).to_string(index=False)); T5.to_csv(OUT + "/sim_gene_bins.csv", index=False)

out("\n" + "=" * 100); out("表 6  真实锚点药（rep 0）与随机抽药（rep 1-30）对比，强行（小模型行均方根 > 1.5）")
S2 = S[S.rms_small > 1.5].assign(kind=lambda x: np.where(x.rep == 0, "anchors", "random"))
out(S2.groupby(["ct", "kind"]).agg(n=("ratio", "size"), n_drugs=("sm", "nunique"), median=("ratio", "median"), min=("ratio", "min"), max=("ratio", "max")).round(3).to_string())

out("\n" + "=" * 100); out("表 7  强行里有哪些药（small，全部重复合并）：出现次数与放大倍数中位数")
t7 = S[S.rms_small > 1.5].groupby(["sm"]).agg(n=("ratio", "size"), n_ct=("ct", "nunique"), rms_small=("rms_small", "median"), ratio=("ratio", "median"), df=("part_df", "median"), sigma=("part_sigma", "median"), effect=("part_coef", "median")).sort_values("rms_small", ascending=False)
out(t7.round(3).to_string()); t7.to_csv(OUT + "/sim_table7_strong_drugs.csv")
open(OUT + "/sim_summary.txt", "w").write("\n".join(lines) + "\n")
