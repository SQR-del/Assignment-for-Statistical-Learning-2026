# 01_linear 线性 / 近邻

**做法**：在紧凑特征（fingerprint+dsig+csig+尺度）上做 OLS / Ridge / BayesianRidge / PLS / KNN；`*_resid` 变体改为预测 y−dsig 的残差再叠加 dsig。

**可选方法**：linear, ridge, bayesridge, pls, knn, ridge_resid, knn_resid（通过 `method=` 传参切换，其余参数透传给对应函数）。

**运行**：`python 01_linear/run.py`（P1+P2 内部验证）；加 `--protocols P1` 只跑 P1；`--export` 生成提交文件。
