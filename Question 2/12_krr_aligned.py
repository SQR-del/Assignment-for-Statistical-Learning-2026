"""12_krr_aligned.py  对齐验证下的局部 KRR（05a，KRR v2）：锚点标签尺度是否影响预测。
做法同 methods/nn_v2_lab/run_aligned.py，只是模型换成局部 KRR（它只用目标细胞的锚点作标签，对锚点尺度最敏感）。
协议 A 中，被当作新细胞的 T/NK 细胞的可见行（15 个锚点 + 2 个阳性对照；KRR 不用阳性对照）换成：
  small  小模型标签（只用这 17 种药 + DMSO 拟合，sim/<细胞>/small_r00，官方流程），不校正 —— 真实任务中的处境
  calib  小模型标签先按自由度换算到大模型尺度（标签 -> t -> 标签），即只校正因素一
  full   官方标签（大模型）—— 完美校正，给出任何尺度校正所能达到的上限
被预测的行始终是官方标签。协议 V 不受影响，不计算。
用法: python3 12_krr_aligned.py      输出: out/krr_aligned_rows.csv  out/krr_aligned_summary.txt
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "methods")); sys.path.insert(0, HERE)
import harness
from common import Table, rms
from q2lib import slp_from_t, tag, NG

OUT = os.path.join(HERE, "out")


def small_labels(ct):
    d = os.path.join(HERE, "sim", tag(ct)); comp = pd.read_csv(d + "/small_r00/compounds.csv")
    A = np.fromfile(d + "/small_r00/slp.f64", dtype="<f8").reshape(len(comp), NG)
    return dict(zip(comp.sm_name, A)), float(pd.read_csv(d + "/small_r00/scalars.csv").df_total[0]), float(pd.read_csv(d + "/full/scalars.csv").df_total[0])


def df_map(y, d1, d2):                                  # 与 run_aligned.py 相同
    t = stats.t.isf(np.clip(10.0 ** (-np.abs(y)) / 2, 1e-300, 0.5), d1)
    return np.sign(y) * np.abs(slp_from_t(t, d2))


tab = Table(); recs = []
for proto, name, hid, qry in harness.folds(tab):
    if proto != "A": continue
    t = int(tab.c_row[qry[0]]); lab, d1, d2 = small_labels(name); hs = set(hid.tolist())
    for mode in ("small", "calib", "full"):
        tm = harness.masked(tab, hid)
        if mode != "full":
            n = 0
            for i in np.flatnonzero(tab.c_row == t):
                if i in hs: continue
                y = lab[tab.comps[tab.d_row[i]]]; tm.Y[i] = df_map(y, d1, d2) if mode == "calib" else y; n += 1
            assert n == 17 or name == "T cells CD8+", n
        P = harness.krr(tm, hid, qry, 2)
        assert np.isfinite(P).all()
        recs += [dict(cell=name, mode=mode, row=int(i), err=float(e)) for i, e in zip(qry, rms(P - tab.Y[qry]))]
    print(name, "done", flush=True)

R = pd.DataFrame(recs); R.to_csv(OUT + "/krr_aligned_rows.csv", index=False)
W = R.pivot_table(index=["cell", "row"], columns="mode", values="err").reset_index()
rng = np.random.default_rng(0); lines = []
def out(s): print(s); lines.append(s)
out("局部 KRR（v2）在协议 A 上的 MRRMSE；差值为负 = 比 small 好；区间为按行配对自助法 95% 区间（10000 次）")
for ct, g in W.groupby("cell"):
    row = "%-20s n=%d  small %.4f  calib %.4f  full %.4f" % (ct, len(g), g.small.mean(), g.calib.mean(), g.full.mean())
    for m in ("calib", "full"):
        d = (g[m] - g.small).to_numpy(); B = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
        lo, hi = np.quantile(B, [.025, .975]); row += " | %s-small %+.4f [%+.4f, %+.4f]" % (m, d.mean(), lo, hi)
    out(row)
open(OUT + "/krr_aligned_summary.txt", "w").write("\n".join(lines) + "\n")
