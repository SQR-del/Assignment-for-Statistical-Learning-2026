# 12 nn_v2 调整后 + I2

与 `08_nn_base_I2` 相同的模型，只改三个数：输出维度 K 64 → 32，训练轮数 400 → 800，dropout 0.3 → 0.2。
选择过程和全部对比见 `../nn_v2_lab/README.md`。`rows.csv`、`scores.json` 取自 `results/nn_v2_lab/c_K32_ep800_drop02/`（同一配置、同一随机种子）。
