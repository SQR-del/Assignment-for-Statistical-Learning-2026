"""11_effect_space_probe.py  探查：在"效应"或"t 值"空间里做跨细胞迁移，是否比直接迁移标签更好。
用法: python3 11_effect_space_probe.py      输出: out/effect_space_probe.txt
三个无参数的基线，都用其他 T/NK 细胞中同一种药的结果来预测目标细胞（不含阳性对照）：
  label   标签的平均（相当于 methods/03_compound_mean）
  t       t 值的平均，再按目标细胞的自由度换算成标签
  effect  效应（logFC）的平均，除以目标细胞自己的"噪声标准差 x 设计因子"得到 t，再换算成标签
          （噪声标准差来自目标细胞的拟合；设计因子取目标细胞其他普通药的逐基因中位数，不使用被预测那一行的信息）
每个基线带一个收缩系数 a（预测 = a x 上述量，在 t 空间或标签空间相乘）。a 在 T/NK 四种细胞上合并选取，再原样用于 B/髓系的 30 个锚点。
注意：T/NK 作目标时，噪声标准差来自包含被预测药物的拟合，略偏乐观；B/髓系的噪声标准差来自真实的训练拟合。
"""
import numpy as np, pandas as pd
from q2lib import *
F = {ct: load_fit(Q2 + "/official_fit/" + tag(ct)) for ct in ALL_CT}
lines = []
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
A = np.round(np.arange(0, 1.55, 0.05), 2)
rec = {}
for c in ALL_CT:
    f = F[c]; df = float(f["scal"].df_total); src = [s for s in TNK if s != c]
    ordi = [d for d in f["compounds"] if d not in PC]; su_all = np.stack([f["su"][f["row"][d]] for d in ordi]); sd = np.sqrt(f["s2post"])
    for k, d in enumerate(ordi):
        ss = [s for s in src if d in F[s]["row"]]
        if len(ss) < 2: continue
        y = f["slp"][f["row"][d]]
        L = np.mean([F[s]["slp"][F[s]["row"][d]] for s in ss], 0); T = np.mean([F[s]["t"][F[s]["row"][d]] for s in ss], 0)
        E = np.mean([F[s]["coef"][F[s]["row"][d]] for s in ss], 0)
        su = np.median(np.delete(su_all, k, 0), 0); Te = E / (sd * su)
        e = dict(zero=np.full(len(A), rms(y)), label=np.array([rms(a * L - y) for a in A]), t=np.array([rms(slp_from_t(a * T, df) - y) for a in A]),
                 effect=np.array([rms(slp_from_t(a * Te, df) - y) for a in A]))
        rec[(c, d)] = e
M = ["zero", "label", "t", "effect"]
tnk = [k for k in rec if k[0] in TNK]
best = {m: A[int(np.argmin(np.mean([rec[k][m] for k in tnk], 0)))] for m in M}
out("在 T/NK 上合并选出的收缩系数 a:", {m: float(best[m]) for m in M if m != "zero"})
out("\n各目标细胞的 MRRMSE（越小越好）。a=1 为不收缩；a* 为上面选出的系数")
out("%-20s %5s %8s | %8s %8s %8s | %8s %8s %8s" % ("目标细胞", "行数", "全零", "标签 a=1", "t a=1", "效应 a=1", "标签 a*", "t a*", "效应 a*"))
i1 = int(np.argmin(np.abs(A - 1)))
for c in ALL_CT + ["T/NK 合计", "B+髓系 合计"]:
    ks = [k for k in rec if k[0] == c] if c in ALL_CT else (tnk if c.startswith("T/NK") else [k for k in rec if k[0] not in TNK])
    g = lambda m, i: np.mean([rec[k][m][i] for k in ks])
    out("%-20s %5d %8.3f | %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f" % (c, len(ks), g("zero", 0), g("label", i1), g("t", i1), g("effect", i1),
        *[g(m, int(np.argmin(np.abs(A - best[m])))) for m in ("label", "t", "effect")]))
out("\n按药物配对比较（a*）：效应法误差 - 标签法误差，负数表示效应法更好；区间为对药物重抽样")
rng = np.random.default_rng(0)
for nm, ks in (("T/NK", tnk), ("B+髓系", [k for k in rec if k[0] not in TNK])):
    for m in ("t", "effect"):
        d = np.array([rec[k][m][int(np.argmin(np.abs(A - best[m])))] - rec[k]["label"][int(np.argmin(np.abs(A - best["label"])))] for k in ks])
        names = np.array([k[1] for k in ks]); u = np.unique(names); bs = []
        for _ in range(2000):
            pick = rng.choice(u, len(u)); bs.append(np.mean(np.concatenate([d[names == x] for x in pick])))
        out("   %-8s %-6s 平均差 %+.4f  95%% 区间 [%+.4f, %+.4f]  更好的行占 %.0f%%" % (nm, m, d.mean(), np.quantile(bs, .025), np.quantile(bs, .975), 100 * (d < 0).mean()))
open(Q2 + "/out/effect_space_probe.txt", "w").write("\n".join(lines) + "\n")
