# 10_autoencoder 非线性目标自编码器

**做法**：先在目标向量上训练自编码器得到非线性低秩隐码（替代线性 TruncatedSVD），再用embedding MLP 预测隐码并解码回基因空间。**结论：不如线性低秩**（见 REPORT_ROUND2.md）。

**关键参数**：`k`（隐码维度）、`hidden`、`standardize`。
