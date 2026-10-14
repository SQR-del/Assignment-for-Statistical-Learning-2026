# 13_final_blend 最终融合

**做法**：把 round-2 的三个去相关强模型按固定权重融合：`0.50·embedding + 0.30·SMILES-CNN + 0.20·hybrid`，各分量多随机种子取中位数，再裁剪到训练范围。权重来自参考折学习/稳健性分析。

**提交**：`python code/predict.py` 用同一配方（更多种子）在全部数据上生成 `results/submission.csv`。
