# 06_mlp_torch 全输出 MLP（torch）

**做法**：紧凑特征 → 逐基因标准化 → 多层 MLP 直接回归 18,211 维；Adam+（可选余弦退火）。`residual=True` 时改为预测 y−dsig 再叠加。

**关键参数**：`hidden`（层宽元组）、`epochs`、`wd`、`residual`。
