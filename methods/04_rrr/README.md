# 04_rrr 降秩回归 RRR

**做法**：把系数矩阵约束为低秩（Y = X B, rank(B)≤r），监督式选取输出子空间，代替输出 PCA。

**可选方法**：rrr（通过 `method=` 传参切换，其余参数透传给对应函数）。

**运行**：`python 04_rrr/run.py`（P1+P2 内部验证）；加 `--protocols P1` 只跑 P1；`--export` 生成提交文件。
