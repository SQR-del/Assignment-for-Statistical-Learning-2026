"""05_checks.py  推导的数值检验（存档版）。
用法: python3 05_checks.py     输出: out/checks.txt
检验 1  标签 = sign(t) * -log10(2 P(T_df > |t|))，df = 残差自由度 + 先验自由度
检验 2  设计因子（含权重）在小模型与大模型之间是否变化
检验 3  一阶近似公式 y_大/y_小 = 1 + ln10 * y * (1/df_小 - 1/df_大) 与精确换算对比
检验 4  全零基因：小模型中全部同值；大模型中同值 <=> 所有孔全零
检验 5  官方标签中的全零基因检验（B/髓系训练标签只用了训练孔）
数据与拟合均为通过逐数值验证的官方流程（见 06_verify_official.py）
"""
import os
import numpy as np, pandas as pd
from scipy import stats
from q2lib import *

OUT = Q2 + "/out"; os.makedirs(OUT, exist_ok=True); lines = []
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

anch = [x.strip() for x in open(Q2 + "/cache_official/anchors.txt")]
out("[检验 1] 标签公式（float32 存盘，容许 1e-4 的相对误差）")
for ct in TNK:
    for nm in ("full", "small_r00"):
        f = load_fit(Q2 + "/sim/%s/%s" % (tag(ct), nm)); d = float(f["scal"].df_total)
        e = np.abs(f["slp"] - slp_from_t(f["t"], d)) / np.maximum(np.abs(f["slp"]), 1)
        out("   %-20s %-10s df %.1f  条目 %d  最大相对误差 %.1e" % (ct, nm, d, e.size, e.max()))

out("\n[检验 2] 设计因子之比 小/大（含 voom 权重，逐基因；15 种锚点药合并）")
for ct in TNK:
    fs = load_fit(Q2 + "/sim/%s/small_r00" % tag(ct)); ff = load_fit(Q2 + "/sim/%s/full" % tag(ct))
    q = np.concatenate([fs["su"][fs["row"][s]] / ff["su"][ff["row"][s]] for s in fs["compounds"] if s not in PC])
    out("   %-20s 中位数 %.3f  5%%-95%% 分位 %.3f-%.3f" % (ct, np.median(q), np.quantile(q, .05), np.quantile(q, .95)))

out("\n[检验 3] 自由度换算 168 -> 424：精确值 与 一阶近似")
for y in (1, 3, 5, 10, 20, 40, 80, 150):
    t = stats.t.isf(10.0 ** (-y) / 2, 168); ex = float(slp_from_t(np.array([t]), 424)[0]) / y
    out("   y=%4d  精确 x%.3f  近似 x%.3f" % (y, ex, 1 + LN10 * y * (1 / 168 - 1 / 424)))

out("\n[检验 4] 我们自己的流程：小模型孔全零的基因")
for ct in TNK:
    fs = load_fit(Q2 + "/sim/%s/small_r00" % tag(ct)); ff = load_fit(Q2 + "/sim/%s/full" % tag(ct))
    zr = zero_mask(ct, list(anch) + list(PC)); za = zero_mask(ct)
    cs = np.all(np.isclose(fs["slp"], fs["slp"][:, [np.flatnonzero(zr)[0]]], rtol=1e-6, atol=1e-6), axis=0)
    cf = np.all(np.isclose(ff["slp"], ff["slp"][:, [np.flatnonzero(za)[0]]], rtol=1e-6, atol=1e-6), axis=0)
    out("   %-20s 小模型孔全零 %4d | 小模型中同值 %4d | 所有孔全零 %4d | 大模型中同值 %4d | 不一致 %d" % (ct, zr.sum(), (zr & cs).sum(), (zr & za).sum(), (zr & cf).sum(), (zr & (cf ^ za)).sum()))

out("\n[检验 5] 官方标签：训练孔全零的基因是否都落在每行同值的基因块中")
de, Y, key = official(); ctv = de.cell_type.astype(str).to_numpy()
def const_set(ct):
    core = None
    for i in np.flatnonzero(ctv == ct):
        v, c = np.unique(np.round(Y[i], 9), return_counts=True); m = np.isclose(Y[i], v[np.argmax(c)], atol=1e-9)
        core = m if core is None else core & m
    return core
for ct in ALL_CT:
    core = const_set(ct); z = zero_mask(ct)
    out("   %-20s 官方同值基因 %4d | 训练孔全零 %4d | 其中同值 %4d (%.0f%%)" % (ct, core.sum(), z.sum(), (z & core).sum(), 100 * (z & core).sum() / z.sum()))
out("   对照：T/NK 上只看 17 种训练药的孔")
for ct in TNK:
    core = const_set(ct); z = zero_mask(ct, list(anch) + list(PC))
    out("   %-20s 限制孔全零 %4d | 其中官方同值 %4d (%.0f%%)" % (ct, z.sum(), (z & core).sum(), 100 * (z & core).sum() / z.sum()))
open(OUT + "/checks.txt", "w").write("\n".join(lines) + "\n")
