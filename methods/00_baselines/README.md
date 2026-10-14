# 00_baselines 平凡基线

**做法**：全局均值 / 药物签名（dsig）/ 细胞类型签名（csig）/ 加性组合。dsig 是跨细胞、排除本细胞类型后目标药物的平均响应；csig 是该细胞类型 15 个锚点药的平均响应。

**可选方法**：global_mean, drug_mean, cell_anchor_mean, additive（通过 `method=` 传参切换，其余参数透传给对应函数）。

**运行**：`python 00_baselines/run.py`（P1+P2 内部验证）；加 `--protocols P1` 只跑 P1；`--export` 生成提交文件。
