"""中期报告第 1 节用到的统计量，以及图 2(b)、图 3 的数据。需要 data/de_train.parquet（服务器上）。
用法: python 1_data_exploration/report_stats.py
输出（outputs/）:
  row_rms.csv      每行标签的均方根和 |y|<1 的比例（图 3，reports/midterm/make_fig.py）
  corr_tnk.csv     同一种药在两种 T/NK 细胞之间响应的平均 Pearson 相关（图 2(b)，reports/midterm/make_fig_corr.py；
                   按 detrain_analysis/ 同一做法：去掉阳性对照，取四种 T/NK 细胞共有的药物，逐药物求相关后平均）
  summary.json     标签分布、各行均方根、锚点与其余药物的强度比较
"""
import os, json, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(os.path.join(ROOT, "data", "de_train.parquet"))
for c in ("cell_type", "sm_name"): df[c] = df[c].astype(str)
genes = [c for c in df.columns if c not in ("cell_type", "sm_name", "sm_lincs_id", "SMILES", "control")]
Y = df[genes].to_numpy(np.float64)
TNK = ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]
S = {}

# 标签分布（不含阳性对照）
drug = ~df.control.to_numpy(bool); a = np.abs(Y[drug])
S["labels"] = dict(frac_abs_lt1=float((a < 1).mean()), median_abs=float(np.median(a)), max=float(Y[drug].max()), min=float(Y[drug].min()))

# 每行的均方根（图 3）
R = pd.DataFrame(dict(cell_type=df.cell_type, sm_name=df.sm_name, control=df.control.astype(bool),
                      rms=np.sqrt((Y ** 2).mean(1)), frac_abs_lt1=(np.abs(Y) < 1).mean(1)))
R.to_csv(os.path.join(OUT, "row_rms.csv"), index=False)
r = R[~R.control]
S["row_rms"] = {c: dict(median=float(r[r.cell_type == c].rms.median()), frac_below_0_75=float((r[r.cell_type == c].rms < 0.75).mean()))
                for c in TNK}
S["row_rms_null_theory"] = float(np.sqrt(2) / np.log(10))

# 跨细胞相关（图 2(b)）
D = df[~df.control]
mats = {c: D[D.cell_type == c].set_index("sm_name")[genes] for c in TNK}
shared = sorted(set.intersection(*[set(m.index) for m in mats.values()]))
order = ["T cells CD4+", "T regulatory cells", "T cells CD8+", "NK cells"]
C = pd.DataFrame(np.eye(4), index=order, columns=order)
for i, j in itertools.combinations(order, 2):
    C.loc[i, j] = C.loc[j, i] = float(np.nanmean([np.corrcoef(mats[i].loc[d].values, mats[j].loc[d].values)[0, 1] for d in shared]))
C.round(4).to_csv(os.path.join(OUT, "corr_tnk.csv"))
S["crosscell_corr_n_drugs"] = len(shared)

# 锚点与其余药物的强度（锚点 = B 细胞中的非对照药物）
anchors = set(D[D.cell_type == "B cells"].sm_name)
strength = D.assign(m=np.abs(Y[drug]).mean(1))
S["anchor_strength_median"] = {c: float(strength[(strength.cell_type == c) & strength.sm_name.isin(anchors)].m.median())
                               for c in TNK + ["B cells", "Myeloid cells"]}
T = strength[strength.cell_type.isin(TNK) & strength.sm_name.isin(shared)]
dm = T.groupby("sm_name").m.max(); is_anc = dm.index.isin(anchors)
rng = np.random.default_rng(0); obs = int((dm[is_anc] > 2).sum())
perm = np.array([(dm.sample(int(is_anc.sum()), random_state=int(rng.integers(1e9))) > 2).sum() for _ in range(20000)])
S["anchors_vs_others_shared_drugs"] = dict(n_anchors=int(is_anc.sum()), n_others=int((~is_anc).sum()),
                                           strong_anchors=obs, strong_others=int((dm[~is_anc] > 2).sum()),
                                           permutation_p=float((perm >= obs).mean()),
                                           note="strong = 四种 T/NK 细胞中最强的平均 |y| > 2；锚点中 Alvocidib、Oprozomib 在 CD8+ 缺测，不在共有药物中")
json.dump(S, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(S, ensure_ascii=False, indent=2))
