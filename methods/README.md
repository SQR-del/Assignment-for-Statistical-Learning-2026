# methods/ 方法族索引

每个子目录是一个方法族：

- `run.py`：调用该方法的预测函数入口（`predict(ctx, **params)`），参数可作为关键字传入；
- `README.md`：做法说明与关键参数；
- `scores.json`：**默认口径 = 队友指标**（`A` / `A15` / `V1` / `V1w` / `objective` + 参数；旧 P1/P2 结果自动保留为 `scores_p1p2.json`）；
- `rows.csv`：逐折的误差明细（`proto,name,row,err`，由 `run.py` 生成）。

## 默认评估：队友指标（A15 / V1w / objective）

从本版起，所有方法默认使用队友 `main` 分支的评估口径（`code/teammate_metric.py`）：

- **A**：依次留出 4 个 T/NK 细胞，只保留其 15 个锚点 + 2 个对照，预测该细胞的官方测试化合物；
  **A15** = 去掉缺 2 个锚点的 CD8+。
- **V1**：B / Myeloid 逐个留出 15 个锚点（30 折）；
  **V1w** = 只统计在来源细胞中响应较弱的锚点（来源均值行 RMS < 官方测试化合物的 90 分位）。
- **objective = mean(V1w / 1.1625, A15 / 0.9894)**，KRR v1 = 1.000，**越低越好**。

共 34 折。预测统一裁剪到训练折可见范围。运行方式：

```bash
python 07_embedding/run.py                    # 默认队友指标（34 折）
python 12_ensemble/run.py                     # 默认队友指标
python 01_linear/run.py --metric p1p2         # 旧 P1/P2 口径
python 09_smiles_cnn/run.py --export          # 全量训练，生成提交文件
```

> 校准：`drug_mean` 在队友口径下得到 A15=1.101、V1w=1.142、objective=1.048，
> 与队友 `main` 的“同化合物均值”基线（1.101 / 1.142 / 1.048）一致。

本版已重跑的三个代表（其余方法运行 `run.py` 即可得到同口径分数）：

| 方法 | A15 | V1w | objective ↓ |
|---|---:|---:|---:|
| 00 `drug_mean`（=dsig） | 1.101 | 1.142 | **1.048** |
| 01 `ridge`（α=10） | 1.206 | 1.645 | 1.317 |
| 12 `ensemble`（α=100, k=10, PLS=10） | 1.569 | 3.245 | 2.188 |

> `12_ensemble` 的 linear-KRR 分量在 A CD8+ 折上仍会数值爆炸（A_CD8≈10.5），
> 新口径把它暴露出来；要用它选模，需要先把该分量正则化/标准化，不能直接沿用 P1 配置。

## 旧 P1 总表（历史对照，官方 mrrmse）

| # | 方法族 | 说明 | 默认配置 | mrrmse ↓ |
|---|---|---|---|---:|
| 00 | `00_baselines` | 全局均值 / dsig / csig / 加性基线 | `drug_mean`（=dsig） | 1.081 |
| 01 | `01_linear` | 紧凑特征上的线性 / PLS / KNN / 残差变体 | `ridge` | 1.080 |
| 02 | `02_kernel` | 核岭（线性/RBF）+ 逐基因锚点迁移 | `krr_linear_sig` | 1.082 |
| 03 | `03_pca_trees` | 目标空间 PCA + Ridge/XGB/RF/MLP、树模型 | `pca_ridge` | 1.050 |
| 04 | `04_rrr` | 降秩回归（监督式低秩输出） | `rank=20` | 1.031 |
| 05 | `05_mlp_sklearn` | sklearn MLP（无 torch） | `(256,128), α=0.01` | 1.052 |
| 06 | `06_mlp_torch` | 全输出 torch MLP | `(512,256), 150ep` | 1.012 |
| 07 | `07_embedding` | **类别 embedding MLP + 逐样本 RMSE** | `emb=128, 秩25` | **0.899** |
| 08 | `08_hybrid` | embedding + dsig(PCA) | `n_extra=64` | 0.937 |
| 09 | `09_smiles_cnn` | SMILES 字符 CNN（自实现公开思路） | `+drug id emb` | 0.904 |
| 10 | `10_autoencoder` | 非线性目标自编码器（无效） | `k=25` | 0.963 |
| 11 | `11_bilinear_cp` | 双线性 / CP 张量分解（无效） | `R=20` | 1.275 |
| 12 | `12_ensemble` | 线性/核族稳健集成 | 嵌套 CV 选参 | 1.003 |
| 13 | `13_final_blend` | **最终融合**（提交配方） | `0.5·embed+0.3·smiles+0.2·hybrid` | ~0.90 |
| 14 | `14_cluster_adapt` | 药物簇 + 化学辅助（重训 / 后处理平滑） | `γ=0.1, λ=0.2` | 见 `REPORT_CLUSTER_ADAPT.md` |

> 00–06、12 的数值由本地 `run.py` 重新运行得到；07–11、13 取自 `results/` 既有实验
> （`round2_*`、`extend_*`）以及 `SUMMARY.md`/`REPORT*.md`。所有数值同为 P1 协议。

## 结论速览

- **有效**：低秩输出、类别 embedding + 逐样本 RMSE、多种子中位、`dsig` 迁移特征、融合。
- **无效**：目标自编码器（非线性低秩）、CP 张量（尺度错配）、化学指纹单独使用、伪标签、零锚点（P2 全部失效）。
