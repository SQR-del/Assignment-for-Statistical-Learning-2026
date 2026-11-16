"""按组员分支自己的提交流程生成 21（sklearn MLP）、22（宽 MLP）的提交文件，用来核对移植。

与分支 methods/harness.py 的 export() 相同：
  * 数据：common_emb.load_de()（去掉 12 行阳性对照）；
  * 每个目标细胞（B、髓系）单独训练，训练行 = 4 种完整细胞的全部药 + 目标细胞的 15 个锚点；
  * 特征：分支 code/predict.py 的 build_target_context（下面逐字拷贝），即 [指纹 PCA | dsig PCA64]，没有 csig；
  * 预测截断到训练标签的逐基因范围；torch 模型用分支的单一种子。
只有写文件一步用本仓库的 Table.write_submission（按 Kaggle 的 (cell_type, sm_name) 行序），不影响预测值。
用法: CUDA_VISIBLE_DEVICES=0 python3 export_branch_flow.py
"""
import os, sys, hashlib
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "code"))
import common_emb as C
import models_emb as ME
sys.modules.setdefault("models", ME)
import nn_emb
from common import Table


# ---- verbatim copy of build_target_context from the branch code/predict.py (commit 3a87515) ----
def build_target_context(df, genes, anchors, target, idmap, fp, cid, did):
    """Training context + test pairs for one Kaggle target cell type (leak-free)."""
    Y = df[genes].to_numpy(np.float32)
    anchors = set(anchors)
    tr = df.index[df.cell_type.isin(C.FULL_CELLS) |
                  ((df.cell_type == target) & (df.drug.isin(anchors)))].tolist()
    ids = idmap.index[idmap.cell_type == target].tolist()
    te_drugs = [idmap.at[i, "sm_name"] for i in ids]
    ytr = Y[tr]

    def drug_sig(cell, drug):
        vs = [Y[df.index[(df.cell_type == o) & (df.drug == drug)]][0]
              for o in C.FULL_CELLS if o != cell
              and len(df.index[(df.cell_type == o) & (df.drug == drug)])]
        return np.mean(vs, 0) if vs else np.zeros(len(genes), np.float32)

    Dtr = np.stack([drug_sig(df.at[i, "cell_type"], df.at[i, "drug"]) for i in tr]).astype(np.float32)
    Dte = np.stack([drug_sig(target, d) for d in te_drugs]).astype(np.float32)
    x_tr = np.array([[cid[df.at[i, "cell_type"]], did[df.at[i, "drug"]]] for i in tr], dtype=np.int64)
    x_te = np.array([[cid[target], did[d]] for d in te_drugs], dtype=np.int64)
    # compact view for the classic ensemble
    fp_lookup = fp.set_index("drug")
    Ftr = fp_lookup.loc[df.loc[tr, "drug"].values].to_numpy(np.float32)
    Fte = fp_lookup.loc[te_drugs].to_numpy(np.float32)
    pd_ = PCA(n_components=min(64, len(tr) - 1), random_state=C.SEED).fit(Dtr)
    def compact(F, D):
        return np.concatenate([F, pd_.transform(D)], 1).astype(np.float32)
    view = dict(Xc_train=compact(Ftr, Dtr), Xc_test=compact(Fte, Dte),
                Dsig_train=Dtr, Dsig_test=Dte, y_train=ytr,
                Csig_train=np.zeros((len(tr), 1), np.float32), Csig_test=np.zeros((len(te_drugs), 1), np.float32))
    return dict(view=view, ytr=ytr, x_tr=x_tr, x_te=x_te, Dtr=Dtr, Dte=Dte,
                te_ids=ids, te_drugs=te_drugs, target=target)
# ---- end of copy ----


def predict(name, view):
    if name == "mlp_sklearn":
        return ME.m_mlp_sklearn(view, hidden=(256, 128), alpha=1e-3)
    Xtr, Xte = ME._scaled(view)
    return nn_emb.train_full_mlp(Xtr, Xte, view["y_train"], hidden=(1024, 512), epochs=150, wd=1e-4,
                                 dsig_tr=view["Dsig_train"], dsig_te=view["Dsig_test"], seed=C.SEED)


df, genes = C.load_de()
for c in ["cell_type", "drug", "SMILES"]: df[c] = df[c].astype(str)      # categorical on disk with the current pandas
anchors = C.anchor_drugs(df); cells, drugs, cid, did = C.cat_indices(df)
fp = C.drug_fingerprint_pca(df, n_comp=128)
idmap = pd.read_csv(C.data_path("id_map.csv"))
tab = Table(); assert list(tab.genes) == list(genes)
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
for folder, name in [("21_mlp_sklearn", "mlp_sklearn"), ("22_mlp_wide", "mlp_wide")]:
    pred = {}
    for target in ["B cells", "Myeloid cells"]:
        ctx = build_target_context(df, genes, anchors, target, idmap, fp, cid, did)
        yp = np.clip(np.asarray(predict(name, ctx["view"]), np.float32), ctx["ytr"].min(0), ctx["ytr"].max(0))
        for d, v in zip(ctx["te_drugs"], yp): pred[(target, d)] = v
    P = np.stack([pred[(c, s)] for c, s in zip(tab.kaggle.cell_type, tab.kaggle.sm_name)]).astype(np.float64)
    out = os.path.join(ROOT, "results", "methods", folder); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "%s_branchflow_kaggle_ids.csv" % folder); tab.write_submission(P, path)
    print("wrote", path, "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest(), flush=True)
