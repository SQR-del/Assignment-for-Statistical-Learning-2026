# 07_embedding 类别 embedding MLP

**做法**：只用 (细胞类型 id, 药物 id) 两个 embedding 作为输入，预测目标向量的低秩压缩（TruncatedSVD 秩 ~25）；损失为逐样本 RMSE（= 官方指标），多随机种子取中位数。这是从公开解法提炼、自行实现的核心单模型（round-2 最强）。

**关键参数**：`emb`、`hidden`、`n_dim`（SVD 秩）、`epochs`、`lr`、`drop`、`reps`（种子数）。
