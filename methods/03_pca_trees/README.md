# 03_pca_trees 目标空间 PCA + 非线性基学习器 / 树模型

**做法**：先把 18211 维目标用 PCA 压到 ~128 维，再上 Ridge/XGBoost/RF/sklearn-MLP（否则树模型过重）；以及直接的多输出树 / 梯度提升（仅在前若干高方差基因上）。

**可选方法**：pca_ridge, pca_xgb, pca_rf, pca_mlp, xgb_multi, random_forest, extra_trees, hgb_multi（通过 `method=` 传参切换，其余参数透传给对应函数）。

**运行**：`python 03_pca_trees/run.py`（P1+P2 内部验证）；加 `--protocols P1` 只跑 P1；`--export` 生成提交文件。
