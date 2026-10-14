# 12_ensemble 稳健集成（线性/核族）

**做法**：对去相关的 5 个族（PCA-Ridge、Ridge、PLS、KNN、线性核岭）取等权平均；默认参数由嵌套 CV 选出（ridge_alpha=100, knn_k=10, pls_n=10）。

**可选方法**：ensemble（通过 `method=` 传参切换，其余参数透传给对应函数）。

**运行**：`python 12_ensemble/run.py`（P1+P2 内部验证）；加 `--protocols P1` 只跑 P1；`--export` 生成提交文件。
