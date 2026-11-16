# 24：第一轮融合

来自组员分支 `singlecell-perturbation-project` 的第一轮提交（分支 REPORT.md §12）。模型代码原样保留在 `code/models_emb.py`、`code/nn_emb.py`，特征由 `code/common_emb.py` 的 `build_views` 构造，通过 `code/classic_adapter.py` 接入本仓库的验证框架。

**做法**：mlp_wide、mlp（(512,256)，其余同 22）、23、21 四个模型的等权平均，即分支 §12 的主提交 `submission.csv`。

**特征**：SMILES 字符 n-gram 指纹的 PCA、药物签名 dsig（同药在其他 T/NK 细胞的平均响应）与细胞签名 csig（本细胞锚点药的平均响应）各自的 PCA，以及两者的幅度，共约 260 维；签名只用训练折可见行。预测截断到训练标签的逐基因范围。torch 模型用分支的单一种子。

**官方分数（组员提交，近似）**：私有 / 公开 0.902 / 0.659。组员提交时训练集不含阳性对照和另一种稀疏细胞的锚点，与这里的训练行不同，所以只作近似参考，没有记入 `results/kaggle_scores.csv`。

`rows.csv`、`scores.json` 由 `../classic_lab/run_all.py` 生成（34 折）。
