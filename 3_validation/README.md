# 3_validation：验证设计（对应中期报告第 3 节）

课程要求不使用测试标签，所以内部验证需要自己构造，并尽量模拟真实任务：目标细胞只给 15 个锚点，预测其余药物。

## 协议（共 34 折）

| 协议 | 做法 | 折数 |
|---|---|---|
| A | 依次将一种 T/NK 细胞（NK、CD4+、CD8+、Treg）视为新细胞，只保留它的 15 个锚点和 2 个阳性对照（CD8+ 只有 13 个锚点），预测它的官方测试药物 | 4 |
| V | 在 B 细胞和髓系细胞上，每次留出一个锚点来预测 | 30 |

- **V1w（弱锚点）**：协议 V 中，在来源细胞里响应较弱的锚点（来源平均行均方根低于官方测试药物的 90% 分位数）。它们更像大多数测试药物。
- **综合指标** = mean(V1w / 1.1625, A / 0.9402)，分母是第一版局部核岭回归（KRR v1，`archive/methods_removed/05a_krr_v1`）的分数，所以 KRR v1 = 1.000，越低越好。
- **口径变更**：2026-11-16 之前，协议 A 不计 CD8+（记为 A15），综合指标为 mean(V1w / 1.1625, A15 / 0.9894)。此前的模型选择和调参都是按旧口径做的；所有 `scores.json` 已按现行口径从 `rows.csv` 重算，主要模型的排序不变。

## 文件
- **`harness.py`**：所有方法共用的验证框架。定义 34 折、计分（`summarize`）、导出提交文件（`--export`），并从 `5_official_scores/kaggle_scores.csv` 读取官方分数写进各方法的 `scores.json`。被留出的行在交给方法之前置为 NaN，方法一旦读取就会报错，从机制上防止测试标签泄漏。`python 3_validation/harness.py` 只刷新各方法的官方分数字段。
- **`validate.py`**：第一版最终模型（`4_models/anchor_transfer/10_blend`）的消融：底座、两项改进、KRR 和融合，结果在 `results/`。
- **`results/`**：`validate.py` 的输出（`rows.csv`、`summary.json`、`run.log`）。

## 相关实验
- **对齐验证**：协议 A 中把锚点换成"只用 15 种药拟合"的小模型标签，使锚点与待预测行的尺度关系与真实任务一致。神经网络见 `4_models/anchor_transfer/nn_v2_lab/run_aligned.py`，局部核岭回归见 `2_label_generation/12_krr_aligned.py`（`2_label_generation/README.md` 第 10 节）。
