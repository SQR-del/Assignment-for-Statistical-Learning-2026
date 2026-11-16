"""09_group_variance.py  汇总 09_group_variance.R，并预测"只因样本构成不同"造成的噪声估计之比。
用法: Rscript 09_group_variance.R 6 && python3 09_group_variance.py      输出: out/group_variance_summary.txt
预测方法：合并残差方差 = 各组方差按残差自由度加权。小模型的普通药只有 15 种（自由度约为 15*2=30），大模型有全部普通药。
  pred_sd_ratio = sqrt(小模型合并方差 / 大模型合并方差)；标签约与 t^2 成正比，故预测的标签之比（大/小）约为 pred_sd_ratio^2。
对 B/髓系：只有训练样本，普通药组就是 15 种锚点药；大模型的普通药方差未知，用锚点药的方差代替（假设测试药的孔与锚点药的孔同样吵）。
"""
import numpy as np, pandas as pd
from q2lib import *
R = pd.read_csv(Q2 + "/out/group_variance.csv"); R = R[(R.v_ORD > 0) & (R.v_DMSO > 0) & (R.v_PC > 0)]
lines = []
def out(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)
rng = np.random.default_rng(0)
out("各组残差方差 / voom 预测方差（3000 个基因的中位数；括号内为对基因重抽样的 95% 区间）及其含义")
obs = {"NK cells": 1.00, "T cells CD4+": 0.91, "T cells CD8+": 1.11, "T regulatory cells": 1.07}      # 模拟中弱行的实际 大/小 之比（表 1）
for ct, g in R.groupby("ct", sort=False):
    def stat(x):
        vd, vp, vo = x.v_DMSO.median(), x.v_PC.median(), x.v_ORD.median(); dd, dp, do = x.df_DMSO.median(), x.df_PC.median(), x.df_ORD.median()
        small = (dd * vd + dp * vp + 30 * vo) / (dd + dp + 30)
        n_ord_full = 429 if ct in ("B cells", "Myeloid cells") else x.n_ORD.iloc[0]
        do_full = do if ct not in ("B cells", "Myeloid cells") else n_ord_full - 143          # 大模型：约 143 种普通药各占 1 个参数
        full = (dd * vd + dp * vp + do_full * vo) / (dd + dp + do_full)
        return vd, vp, vo, vd / vo, vp / vo, np.sqrt(small / full), small / full
    s = stat(g); B = np.array([stat(g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(300)])
    lo, hi = np.quantile(B, [.025, .975], axis=0)
    out("%-20s DMSO %.2f  PC %.2f  ORD %.2f | DMSO/ORD %.2f (%.2f-%.2f)  PC/ORD %.2f (%.2f-%.2f) | 预测噪声标准差 小/大 %.3f，预测标签 大/小 %.2f (%.2f-%.2f)%s" %
        (ct, s[0], s[1], s[2], s[3], lo[3], hi[3], s[4], lo[4], hi[4], s[5], s[6], lo[6], hi[6], (" | 模拟实测（弱行）%.2f" % obs[ct]) if ct in obs else " | 无法实测"))
open(Q2 + "/out/group_variance_summary.txt", "w").write("\n".join(lines) + "\n")
