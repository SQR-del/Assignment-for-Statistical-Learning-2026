"""q2lib.py  读取拟合结果、标签与 t 值互换的公共函数。"""
import os
import numpy as np, pandas as pd
from scipy import stats, special

P = "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction"
Q2 = P + "/2_label_generation"
LN10 = np.log(10.0)
DMSO = "Dimethyl Sulfoxide"; PC = ("Belinostat", "Dabrafenib")
TNK = ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]
ALL_CT = ["B cells", "Myeloid cells"] + TNK
GENES = [g.strip() for g in open(Q2 + "/cache_official/genes_de_train_order.txt")]
NG = len(GENES)


def tag(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def load_fit(d, light=False):
    """读一个拟合目录。矩阵形状为 (药物数, 基因数)。"""
    comp = pd.read_csv(d + "/compounds.csv"); K = len(comp)
    rd = lambda nm, shape: np.fromfile(d + "/" + nm, dtype="<f4").astype(np.float64).reshape(shape)
    f = dict(compounds=list(comp.sm_name), wells=list(comp.wells), scal=pd.read_csv(d + "/scalars.csv").iloc[0],
             slp=rd("slp.f32", (K, NG)), t=rd("t.f32", (K, NG)))
    if not light and os.path.exists(d + "/coef.f32"):
        f.update(coef=rd("coef.f32", (K, NG)), su=rd("su.f32", (K, NG)), s2post=rd("s2post.f32", (NG,)), sigma=rd("sigma.f32", (NG,)))
    f["row"] = {s: i for i, s in enumerate(f["compounds"])}
    return f


def log_sf_t(t, df):
    """t 分布上尾概率的对数；极端尾部（scipy 下溢处）改用不完全 beta 函数的级数。"""
    t = np.abs(np.asarray(t, dtype=np.float64)); out = stats.t.logsf(t, df)
    bad = ~np.isfinite(out) | (out < -650)
    if bad.any():
        tb = t[bad]; a, b = df / 2.0, 0.5; x = df / (df + tb * tb)
        s = np.ones_like(tb); term = np.ones_like(tb)
        for k in range(400):                      # 2F1(a+b, 1; a+1; x)
            term = term * x * (a + b + k) / (a + 1 + k); s += term
        out[bad] = np.log(0.5) + a * np.log(x) + b * np.log1p(-x) - np.log(a) - special.betaln(a, b) + np.log(s)
    return out


def slp_from_t(t, df):
    """t 值 -> 标签：sign(t) * -log10(2 * P(T_df > |t|))"""
    return np.sign(t) * -(log_sf_t(t, df) + np.log(2.0)) / LN10


def official():
    de = pd.read_parquet(P + "/data/de_train.parquet")
    Y = de[GENES].to_numpy(np.float64)
    key = {(c, s): i for i, (c, s) in enumerate(zip(de.cell_type.astype(str), de.sm_name.astype(str)))}
    return de, Y, key


_CNT = {}
def zero_mask(ct, keep=None):
    """该细胞类型的样本（可限定保留哪些药 + DMSO）中计数全为 0 的基因，按 de_train 的基因顺序。数据：官方步骤重建的伪 bulk。"""
    import scipy.sparse as sp
    if not _CNT:
        S = pd.read_csv(Q2 + "/cache_official/samples.csv"); g = [x.strip() for x in open(Q2 + "/cache_official/genes.txt")]
        gi = {x: i for i, x in enumerate(g)}
        _CNT.update(S=S, C=sp.load_npz(Q2 + "/cache_official/counts_samples_x_genes.npz").tocsc()[:, [gi[x] for x in GENES]].tocsr())
    S = _CNT["S"]; m = (S.cell_type == ct)
    if keep is not None: m &= S.sm_name.isin(list(keep) + [DMSO])
    return np.asarray(_CNT["C"][np.flatnonzero(m.to_numpy())].sum(0)).ravel() == 0
