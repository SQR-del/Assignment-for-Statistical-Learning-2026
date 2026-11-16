# 19：分支的融合

来自组员分支 `singlecell-perturbation-project` 的 `methods/13_final_blend`，模型代码原样保留在 `code/nn_emb.py`，通过 `code/emb_adapter.py` 接入本仓库的验证框架。

**做法**：0.5 × 16 + 0.3 × 18 + 0.2 × 17，权重固定，沿用分支的设定。

**共同设置**：输出为 25 维截断 SVD 表示，损失为按行的均方根误差，Adam，400 轮，学习率 3e-3，dropout 0.2；多个随机种子取中位数；预测截断到训练标签的逐基因范围。超参数沿用分支调好的值，没有在本仓库的验证上重新调整。

`rows.csv`、`scores.json` 由 `../emb_lab/run_all.py` 生成（34 折，3 个随机种子，与其他方法相同的折和被预测行）。移植的核对与说明见 `../emb_lab/README.md`。
