# methods/ 方法族索引

每个子目录是一个方法族：

- `run.py`：调用该方法的预测函数入口（`predict(ctx, **params)`），参数可作为关键字传入；
- `README.md`：做法说明与关键参数；
- `scores.json`：内部验证分数（P1/P2 的 mrrmse/MSE/R²/Pearson）与所用参数；
- `rows.csv`：逐折的指标明细（由 `run.py` 生成）。

所有方法都在**同一套泄漏安全的协议**上评估（P1 主协议，见根 `README.md`），
指标以官方 **mrrmse**（mean row-wise RMSE，越小越好）为准。运行方式：

```bash
python 07_embedding/run.py --protocols P1     # 单方法，只跑 P1
python 12_ensemble/run.py                     # P1 + P2
python 09_smiles_cnn/run.py --export          # 全量训练，生成提交文件
```

## P1 总表（默认配置，4 折均值，官方 mrrmse）

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

> 00–06、12 的数值由本地 `run.py` 重新运行得到；07–11、13 取自 `results/` 既有实验
> （`round2_*`、`extend_*`）以及 `SUMMARY.md`/`REPORT*.md`。所有数值同为 P1 协议。

## 结论速览

- **有效**：低秩输出、类别 embedding + 逐样本 RMSE、多种子中位、`dsig` 迁移特征、融合。
- **无效**：目标自编码器（非线性低秩）、CP 张量（尺度错配）、化学指纹单独使用、伪标签、零锚点（P2 全部失效）。
