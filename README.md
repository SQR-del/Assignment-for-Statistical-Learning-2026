# 统计学习大作业：跨细胞类型的药物响应预测（NeurIPS 2023 单细胞扰动）

任务：已知药物在 NK 和三类 T 细胞中的差异表达标签（signed −log10 p，18,211 个基因），以及 B 细胞和髓系细胞中 15 个锚点药的标签，预测 B 细胞和髓系细胞其余 255 个"细胞类型 × 药物"组合。评价指标为 MRRMSE（逐行 RMSE 的均值）。

## 当前结果

最终模型是**融合模型**（`4_models/anchor_transfer/15_blend_v2`）：

```
0.4 · (神经网络 + I2 幅度输出头) + 0.4 · (神经网络 + I1 + I2) + 0.2 · 局部核岭回归
```

- 内部验证综合指标 0.925（第一版核岭回归 = 1，越低越好）。
- 官方分数 private 0.734 / public 0.573（赛后提交；私有榜第一名 0.729，中位数 0.793）。
- 模型选择只用内部验证，没有使用测试标签或排行榜。

## 目录（按中期报告的章节组织）

| 目录 | 报告章节 | 内容 |
|---|---|---|
| `1_data_exploration/` | 1 任务与数据 | 探索性分析的主要发现，以及各图和完整分析的位置 |
| `2_label_generation/` | 2 标签生成机制 | 复现官方 limma 流程（与官方标签逐数值一致）；训练与测试标签的尺度差异；锚点尺度校正的检验 |
| `3_validation/` | 3 验证设计 | 协议 A、V 与综合指标；所有方法共用的验证框架 `harness.py` |
| `4_models/` | 4 模型与结果 | 按类别组织的 26 个方法：`baselines/`、`classical/`、`anchor_transfer/`、`embedding/`；总表在 `4_models/README.md` |
| `5_official_scores/` | 5 内部验证与官方评分的一致性 | 全部 Kaggle 提交记录及一致性分析 |
| `src/` | — | 各方法共用的模型代码 |
| `reports/` | — | 开题报告（`proposal/`）、中期报告（`midterm/`，含画图脚本） |
| `archive/` | — | 已移除的方法和重组前的历次实验记录 |
| `tools/` | — | 2026-11-16 目录重组用的脚本 |

## 运行

完整项目（数据、提交文件、中间结果）在服务器 pku-server 上：

```text
/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/
```

代码需要服务器上的数据和 GPU。在项目根目录运行，例如：

```bash
python 4_models/anchor_transfer/15_blend_v2/run.py            # 内部验证
python 4_models/anchor_transfer/15_blend_v2/run.py --export   # 导出提交文件
```

只在服务器上、不进入 git 的内容：
- `data/`：原始数据，约 3.8 GB。
- `results/`：各方法导出的提交文件（`results/methods/`）、移植实验的逐折预测（`results/emb_lab/`、`results/classic_lab/`）等生成的输出。
- `submissions/`：实际上传到 Kaggle 的文件。
- `2_label_generation/` 下的数据缓存和拟合结果（`cache_official/`、`official_fit/`、`sim/`、`weights_exp/`，约 4 GB）。
- `archive/` 中的旧 LGT 代码与结果。
