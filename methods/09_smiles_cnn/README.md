# 09_smiles_cnn SMILES 字符 CNN

**做法**：自实现公开的“NLP regression”思路——对 SMILES 逐字符 embedding，1-D 卷积 + 全局max-pool 得到分子向量，与细胞类型 embedding 拼接后预测低秩目标。与固定指纹互补。

**关键参数**：`use_drug_emb`（是否叠加药物 id embedding）、`n_dim`、`reps`。
