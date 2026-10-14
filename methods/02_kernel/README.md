# 02_kernel 核方法 / 逐基因锚点迁移

**做法**：紧凑特征或高维 dsig/csig 上的核岭回归（线性 / RBF）；`pg_anchor_ridge` 逐基因用目标细胞的 15 个锚点药学习 参考签名→目标 的映射。

**可选方法**：krr_rbf, krr_linear_sig, krr_rbf_sig, pg_anchor_ridge（通过 `method=` 传参切换，其余参数透传给对应函数）。

**运行**：`python 02_kernel/run.py`（P1+P2 内部验证）；加 `--protocols P1` 只跑 P1；`--export` 生成提交文件。
