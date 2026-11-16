# 23：第一轮经典集成

来自组员分支 `singlecell-perturbation-project` 的第一轮提交（分支 REPORT.md §12）。模型代码原样保留在 `src/models_emb.py`、`src/nn_emb.py`，特征由 `src/common_emb.py` 的 `build_views` 构造，通过 `src/classic_adapter.py` 接入本仓库的验证框架。

**做法**：pca_ridge(α=10) + ridge(α=10) + PLS(20) + 核岭 RBF[dsig,csig](α=1) + KNN(k=5，距离加权) 的等权平均。组成按分支 REPORT.md §5 的表推断（分支现有代码里的 `ensemble` 已是 §13 调参后的另一版本）。

**特征**：SMILES 字符 n-gram 指纹的 PCA、药物签名 dsig（同药在其他 T/NK 细胞的平均响应）与细胞签名 csig（本细胞锚点药的平均响应）各自的 PCA，以及两者的幅度，共约 260 维；签名只用训练折可见行。预测截断到训练标签的逐基因范围。torch 模型用分支的单一种子。

**官方分数（组员提交，近似）**：私有 / 公开 0.885 / 0.645。组员提交时训练集不含阳性对照和另一种稀疏细胞的锚点，与这里的训练行不同，所以只作近似参考，没有记入 `5_official_scores/kaggle_scores.csv`。

`rows.csv`、`scores.json` 由 `../classic_lab/run_all.py` 生成（34 折）。
