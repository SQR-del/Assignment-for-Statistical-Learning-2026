# 13：底座 + I2（nn_v2 设置）

与 `08_nn_base_I2` 相同的模型，代码用 `src/nn_v2.py`，只改三个数：输出维度 K 64 → 32，训练轮数 400 → 800，dropout 0.3 → 0.2。
设置是在"底座 + I2"上用内部验证选出的，然后原样用于其余变体；选择过程见 `../nn_v2_lab/README.md`。
`rows.csv`、`scores.json` 取自 `4_models/anchor_transfer/nn_v2_lab/runs/c_K32_ep800_drop02/`（同一配置、同一随机种子）。提交文件由 `../nn_v2_lab/run_export.py base_I2` 生成。
