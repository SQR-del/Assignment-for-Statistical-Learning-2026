# 12：底座 + I1（nn_v2 设置）

与 `07_nn_base_I1` 相同的模型，代码用 `code/nn_v2.py`，只改三个数：输出维度 K 64 → 32，训练轮数 400 → 800，dropout 0.3 → 0.2。
设置是在"底座 + I2"上用内部验证选出的，然后原样用于其余变体；选择过程见 `../nn_v2_lab/README.md`。
`rows.csv`、`scores.json` 取自 `results/nn_v2_lab/v2_base_I1/`（同一配置、同一随机种子）。提交文件由 `../nn_v2_lab/run_export.py base_I1` 生成。
