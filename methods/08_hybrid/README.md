# 08_hybrid 混合模型（ids + dsig [+ 基线表达]）

**做法**：embedding 捕捉类别身份，dsig（跨细胞药物签名，PCA 降维）提供可迁移的药物效应，一起送入网络预测低秩目标。兼顾“身份”与“跨细胞迁移”。

**关键参数**：`n_extra`（dsig PCA 维度）、`emb`、`n_dim`、`reps`。
