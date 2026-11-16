"""12b_krr_aligned_strong.py  在"理论上会受影响的位置"上评估 12_krr_aligned.py 的三种锚点版本（small / calib / full）。
整体 MRRMSE 由绝大多数弱元素主导，而尺度差异（因素一）只作用于强元素。这里只在以下位置计分：
  S5 / S10  待预测行中真值 |y| > 5 / > 10 的元素
  G         校正实际改动了锚点的基因（任一锚点 |calib - small| > 0.5）在待预测行中的全部元素
指标（每种细胞、每种版本，对所选元素合并计算）：
  rmse      均方根误差
  under     沿真值方向的平均误差 mean((pred - y) * sign(y))，负值 = 幅度偏小；理论预期 small 最负，calib / full 应向 0 靠近
差值相对 small；区间按"待预测行"整行配对自助法（10000 次），因为强元素集中在少数强药物上。
用法: python3 12b_krr_aligned_strong.py      输出: out/krr_aligned_strong.txt
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "methods")); sys.path.insert(0, HERE)
import harness
from common import Table
src = open(os.path.join(HERE, "12_krr_aligned.py"), encoding="utf-8").read()
exec(src.split("tab = Table(); recs = []")[0])          # 复用 small_labels、df_map（不重复运行 12 的主体）

tab = Table(); rng = np.random.default_rng(0); lines = []
def out(s): print(s, flush=True); lines.append(s)
out(__doc__.split("用法")[0].strip()); out("")
MODES = ("small", "calib", "full")
for proto, name, hid, qry in harness.folds(tab):
    if proto != "A": continue
    t = int(tab.c_row[qry[0]]); lab, d1, d2 = small_labels(name); hs = set(hid.tolist())
    P, Anc = {}, {}
    for mode in MODES:
        tm = harness.masked(tab, hid)
        if mode != "full":
            for i in np.flatnonzero(tab.c_row == t):
                if i in hs: continue
                y = lab[tab.comps[tab.d_row[i]]]; tm.Y[i] = df_map(y, d1, d2) if mode == "calib" else y
        P[mode] = harness.krr(tm, hid, qry, 2)
        Anc[mode] = np.stack([tm.Y[tab.row_of[(t, a)]] for a in tab.ANC if (t, a) in tab.row_of and tab.row_of[(t, a)] not in hs])
    Y = tab.Y[qry]; sg = np.sign(Y)
    gchg = np.abs(Anc["calib"] - Anc["small"]).max(0) > 0.5
    masks = {"S5": np.abs(Y) > 5, "S10": np.abs(Y) > 10, "G": np.broadcast_to(gchg, Y.shape)}
    out("%s：待预测行 %d；S5 元素 %d（分布在 %d 行），S10 元素 %d（%d 行），G 基因 %d 个" % (
        name, len(qry), masks["S5"].sum(), (masks["S5"].sum(1) > 0).sum(), masks["S10"].sum(), (masks["S10"].sum(1) > 0).sum(), gchg.sum()))
    for key, M in masks.items():
        if M.sum() == 0: continue
        # 每行的平方误差和、沿真值方向误差和、元素数，供整行重抽样
        se = {m: (((P[m] - Y) ** 2) * M).sum(1) for m in MODES}; ue = {m: (((P[m] - Y) * sg) * M).sum(1) for m in MODES}; cnt = M.sum(1)
        stat = lambda idx, m: (np.sqrt(se[m][idx].sum() / cnt[idx].sum()), ue[m][idx].sum() / cnt[idx].sum())
        base = {m: stat(np.arange(len(qry)), m) for m in MODES}
        rows = np.flatnonzero(cnt > 0) if key != "G" else np.arange(len(qry))
        B = {m: [] for m in ("calib", "full")}
        for _ in range(10000):
            idx = rows[rng.integers(0, len(rows), len(rows))]
            s0 = stat(idx, "small")
            for m in B: s = stat(idx, m); B[m].append((s[0] - s0[0], s[1] - s0[1]))
        txt = "  %-3s small rmse %.3f under %+.3f" % (key, *base["small"])
        for m in ("calib", "full"):
            b = np.array(B[m]); lo, hi = np.quantile(b, [.025, .975], axis=0)
            txt += " | %s rmse %+.3f [%+.3f,%+.3f] under %+.3f [%+.3f,%+.3f]" % (
                m, base[m][0] - base["small"][0], lo[0], hi[0], base[m][1] - base["small"][1], lo[1], hi[1])
        out(txt)
open(os.path.join(HERE, "out", "krr_aligned_strong.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
