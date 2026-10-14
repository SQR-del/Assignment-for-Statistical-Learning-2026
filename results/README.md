# results/ 结果索引

本目录汇总历次实验的**指标 CSV、图、预测与提交文件**。原始数据与超大的提交/预测文件
已在 `.gitignore` 中排除（见下）。

## 主要内容

| 路径 | 内容 |
|---|---|
| `benchmark_raw.csv` / `benchmark_summary.csv` | `code/validate.py` 的最新基准结果（逐折 / 汇总） |
| `metrics_raw.csv`（+ 时间戳副本） | 早期 `run_benchmark` 的逐折指标 |
| `metrics_summary.md` | 由上表生成的 markdown 汇总 |
| `sanity_checks.txt` | 无泄漏/正确性自检输出（`code/validate.py --sanity`） |
| `round2_*_{raw,mean}.csv` | 第二轮：embedding 调参、SMILES-CNN、目标自编码器、CP 张量、稳健性 |
| `extend_*_{raw,mean}.csv` | 数据驱动方向、混合模型、基线特征、目标加权、学习式融合权重 |
| `improve_*_{raw,mean}.csv` | embedding / 公开思路（先验融合、TE-Ridge+TSVD）的对比 |
| `nested_*` / `refcv_*` / `robustness_*` / `robust_choice_*` | 嵌套 CV 调参与参考折稳健性分析 |
| `rrr_results.csv` / `stack_results.csv` / `combo_results.csv` / `ref_combination.csv` / `ablation.csv` | 降秩回归、参考细胞组合、特征消融 |
| `tuning.csv` / `pseudo_offline.csv` / `raw_nn_results.csv` | 早期调参、伪标签离线验证、原始高维输入 NN |
| `control_baseline.csv` | 单细胞 control 细胞的 pseudo-bulk 基线表达（`compute_baseline`） |
| `figures/` | 报告用图（R²-by-model、热图、预测-真值、消融、RRR、参考选择等） |
| `predictions/` | 逐折预测 `P1__<cell>.npz`（**已 gitignore**，可由 `run_benchmark` 重生成） |
| `submission*.csv` | 各版本提交文件（**已 gitignore**，可用 `code/predict.py` 重生成） |
| `run.log` | 早期基准运行日志 |

## 被 gitignore 的大文件

- `results/submission*.csv`：27 个全精度提交文件，约 1.3 GB。
- `results/predictions/`：8 个逐折预测 npz，约 0.7 GB。
- `results/_inner_comp.npz`：学习式权重的中间缓存。

重新生成方式：

```bash
python code/validate.py               # 刷新 benchmark_* 与（可选）预测
python code/predict.py                # 重新生成 results/submission.csv
```

详细实验叙述见项目根目录 `SUMMARY.md`、`REPORT.md`、`REPORT_ROUND2.md`。
