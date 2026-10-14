# 05_mlp_sklearn sklearn MLP

**做法**：紧凑特征 → 逐基因标准化 → sklearn MLPRegressor 回归全部 18211 基因（无需 torch）。

**可选方法**：mlp_sklearn（通过 `method=` 传参切换，其余参数透传给对应函数）。

**运行**：`python 05_mlp_sklearn/run.py`（P1+P2 内部验证）；加 `--protocols P1` 只跑 P1；`--export` 生成提交文件。
