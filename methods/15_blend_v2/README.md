# 15：blend v2（nn_v2 设置）

与 `10_blend` 相同的融合（0.4 × 底座+I2，0.4 × 底座+I1+I2，0.2 × 局部 KRR），两个网络改用 `13`、`14` 的设置（K=32，800 轮，dropout 0.2）。
`rows.csv`、`scores.json` 取自 `results/nn_v2_lab/blend_C/`。提交文件由 `../nn_v2_lab/run_export.py blend` 生成。选择过程见 `../nn_v2_lab/README.md`。
