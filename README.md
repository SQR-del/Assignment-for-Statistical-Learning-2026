# 统计学习大作业：单细胞扰动响应跨细胞类型预测

课程项目 · 数据：**NeurIPS 2023 Open Problems – Single-Cell Perturbations**。
目标：预测「一种药物在一个细胞类型中的全基因组差异表达响应」（标签为 Limma 有符号
`−log10 p`，18,211 个基因），并在官方 test 标签不公开的条件下构造可复现的离线验证。

> 详细实验记录见 [`SUMMARY.md`](SUMMARY.md)、[`REPORT.md`](REPORT.md)、[`REPORT_ROUND2.md`](REPORT_ROUND2.md)。
> 数据研究（EDA、Limma 复现、标签噪声、误差分解）的结论与代码全部合并到
> [`data_analysis/README.md`](data_analysis/README.md)（复现入口 `data_analysis/run_all.py`）。
> 药物簇 + 化学辅助的三模型调整与提交见 [`REPORT_CLUSTER_ADAPT.md`](REPORT_CLUSTER_ADAPT.md)（代码 `methods/14_cluster_adapt/`，提交 `results/submission_cluster.csv`）。
> 历史提交（embedding / PCA+Ridge / sklearn-MLP / wide-MLP / ensemble / 第一轮融合）的逐条复现代码与对齐说明见 [`repro/README.md`](repro/README.md)。

## 目录结构

| 目录 | 内容 | 入口 |
|---|---|---|
| `code/` | 核心库：数据/特征/指标、经典模型、神经网络、融合、提交 | `code/README.md`、`code/predict.py` |
| `methods/` | 各方法族（一目录一方法）：`run.py` + `README.md` + `scores.json`（+ `rows.csv`） | `methods/README.md`（总表） |
| `results/` | 历次实验的指标 CSV、图 `figures/`、预测 `predictions/`、提交文件 | `results/README.md` |
| `data_analysis/` | 数据研究：EDA、Limma 复现、标签噪声、误差分解（结论 + 可复现代码） | `data_analysis/README.md`、`data_analysis/run_all.py` |

## 数据准备（必须）

原始竞赛数据**不随仓库提供**（体积大且受竞赛条款约束）。请从 Kaggle 下载后放到
`<本目录>/data/`，使目录形如：

```
data/
├─ de_train.parquet        # 主表：cell_type × drug，18,211 基因标签
├─ adata_train.parquet     # 单细胞长表：Limma 复现/噪声分析必需
├─ adata_obs_meta.csv      # 单细胞元数据：Limma 复现/噪声分析必需
├─ id_map.csv              # 官方测试映射（B cells 128 + Myeloid 127）
└─ sample_submission.csv   # 提交格式模板
```

若数据在别处，设置环境变量 `SCP_DATA_DIR` 指向该文件夹即可（见 `code/common.py`）。
只用建模/提交时 `adata_*` 可省略；运行 `data_analysis/` 的 Limma 部分则需要。

## 环境

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
python code/validate.py --sanity                 # 无泄漏 / 正确性自检
python code/validate.py --methods ridge ensemble  # 队友指标（默认，34 折）
python methods/07_embedding/run.py                # 单方法，队友指标
python code/validate.py --metric p1p2 --protocols P1 --methods ridge ensemble  # 旧 P1
python code/predict.py                            # 生成 results/submission.csv
```

## 验证协议

- **P1（主）**：目标细胞类型只给 **15 个锚点药**，训练=其余 3 个完整细胞类型的全部药 +
  目标的锚点药，测试=目标的其余 ~129 个药（这些药在别的细胞类型里被观测过）。对应真实
  Kaggle 任务（B / Myeloid）。
- **P2（压力测试）**：目标细胞零锚点，测试其全部药——检验“无锚点能否迁移”。
- **官方指标 `mrrmse`**（mean row-wise RMSE，越小越好）；同时报告 MSE / R² / Pearson。
- **模型选模指标（队友口径，默认）**：34 折 = **A**（依次留出 4 个 T/NK 细胞，只保留其 15 个锚点 + 2 个对照；A15 去掉缺锚点的 CD8+）+ **V1**（B/Myeloid 逐个留出 15 个锚点；V1w 只看在来源细胞中响应较弱的锚点）。
  `objective = mean(V1w / 1.1625, A15 / 0.9894)`，KRR v1 = 1.000，**越低越好**。
  实现见 `code/teammate_metric.py`；旧 P1/P2 协议保留，用 `--metric p1p2` 调用。
  该口径与队友 `main` 分支一致，且本仓库的 `drug_mean` 基线复现出 A15=1.101、V1w=1.142、objective=1.048，与队友的“同化合物均值”基线一致。

> 注意：下文“主要结论”表仍是旧 P1 口径；新口径下部分旧模型会明显变差（例如 ensemble 的 linear-KRR 分量在 A 折上数值不稳定），需要按新 objective 重新调参。

## 主要结论（P1，官方 mrrmse）

| 模型 | mrrmse | 备注 |
|---|---|---|
| **round-2 融合**（最终提交配方） | **~0.90** | `0.50·embedding + 0.30·SMILES-CNN + 0.20·hybrid` |
| embedding（类别 embedding + 逐样本 RMSE，秩 25，多种子中位） | 0.899–0.949 | 最强单模型 |
| hybrid（ids + dsig） | 0.937–0.948 | 兼顾身份与跨细胞迁移 |
| learned-weight 融合 | 0.940 | 参考折学习权重 |
| 线性/核/近邻稳健集成 | 1.003 | 稳健但 mrrmse 稍差 |
| mlp（全输出） | 1.03 | 明显弱于 embedding |
| CP 张量 / 目标自编码器 | 1.28 / 0.96 | 无效或不如线性低秩 |
| 全局均值 | ~1.08 | 基线 |

- **`dsig`（跨细胞药物签名）是主导可迁移特征**；化学指纹单独几乎无用。
- **P2（零锚点）所有模型 R²≈0** → 目标细胞的少量观测是迁移的必需条件。
- **R²/MSE 与 mrrmse 排序不一致**：本项目选模、选参一律以官方 `mrrmse` 为准。

## 复现入口

```bash
python data_analysis/run_all.py    # 数据研究全流程（EDA + Limma + 噪声/误差分解）
python code/validate.py            # 队友指标（A15/V1w/objective），经典模型 + 若干 torch 模型
python code/validate.py --metric p1p2   # 旧 P1/P2 基准
python code/predict.py             # 最终提交（多随机种子融合）
python methods/<族>/run.py         # 单方法族验证（默认队友指标）/ --export 生成提交
```
