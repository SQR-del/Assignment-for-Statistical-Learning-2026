"""06_verify_official.py  把我们按主办方设置算出的标签与 de_train.parquet 逐数值比对。
用法: Rscript 06_verify_official.R && python3 06_verify_official.py      输出: out/verify_official.txt
"""
import os
import numpy as np, pandas as pd
from q2lib import *
OUT = Q2 + "/out"; os.makedirs(OUT, exist_ok=True); lines = []
de, Y, key = official()
tot_ok = True
for ct in ALL_CT:
    d = Q2 + "/official_fit/" + tag(ct); comp = pd.read_csv(d + "/compounds.csv"); sc = pd.read_csv(d + "/scalars.csv").iloc[0]
    A = np.fromfile(d + "/slp.f64", dtype="<f8").reshape(len(comp), NG)
    idx = [key[(ct, s)] for s in comp.sm_name]; O = Y[idx]
    ad = np.abs(A - O); rel = ad / np.maximum(np.abs(O), 1e-12); close = np.allclose(A, O); tot_ok &= close
    r = [np.corrcoef(a, o)[0, 1] for a, o in zip(A, O)]
    lines.append("%-20s 行数 %3d (官方 %3d) | 自由度 残差 %.0f + 先验 %.2f | np.allclose %s | 最大绝对差 %.2e | 最大相对差 %.2e | 每行相关系数最小值 %.10f" %
                 (ct, len(comp), int((de.cell_type.astype(str) == ct).sum()), sc.df_resid, sc.df_prior, close, ad.max(), rel[np.abs(O) > 1e-6].max(), min(r)))
lines.append("全部细胞类型逐数值一致: %s" % tot_ok)
print("\n".join(lines)); open(OUT + "/verify_official.txt", "w").write("\n".join(lines) + "\n")
