# 单细胞扰动响应跨细胞类型预测 —— 总结文档

> 课程项目 · 数据：NeurIPS 2023 Open Problems – Single-Cell Perturbations
> 目标：在**官方 test 标签不公开**的条件下，构造可复现的离线验证，系统比较多种回归 / 机器学习 / 神经网络方法，
> 预测「一种药物在一个细胞类型中的全基因组响应」，并生成 Kaggle 提交。
> 代码与结果见 `exp/`（`REPORT.md` 为逐节详细版，本文件为整体总结）。

---

## 一、数据概述与准备

### 1.1 数据来源与形状
| 文件 | 说明 |
|---|---|
| `data/de_train.parquet` | **主文件**：614 行 = cell_type × (drug/control)，5 元数据列 + **18,211 基因** |
| `data/id_map.csv` | 官方测试映射：255 行（B cells 128 + Myeloid 127） |
| `data/sample_submission.csv` | 提交格式：`id` + 18,211 基因 |
| `data/adata_train.parquet` | 原始单细胞长表：**416,442,312 行**（obs_id/gene/count/normalized_count），240,090 细胞（含 78,867 control） |
| `data/multiome_train.parquet` | multiome baseline（未使用） |

**标签含义**：基因列是 Limma 差异表达统计量 **signed −log10(p) = −log10(p)·sign(LFC)**，
同时编码方向与显著性，**不是** log fold change。长尾：81.5% 的 |值|<1，中位≈0.37，最大 **180.5**。

### 1.2 数据结构关键事实（EDA 核验）
- 6 个细胞类型；4 个"完整"（T CD4+ / T reg / T CD8+ / NK，144 药，CD8+ 为 140），
  **B cells / Myeloid 仅 15 个药**（记作**锚点药集 A**）。
- `144 = 15 (A) ∪ 129 (测试药)`，两组**互斥**；`id_map` 的 129 个测试药**全部**在 4 个完整细胞类型中被观测过。
  → 竞赛缺的正是**新细胞类型**，药物本身在别处有响应。
- **目标矩阵极低秩**：第 1 主成分解释 **65.5%**，10 个 85.4%，20 个 89.5%，90% 只需 23 个（有效维度 ~20–100）。
- 响应强度因细胞类型而异：B/Myeloid 的 mean|v|≈1.8，T 系≈0.6。

### 1.3 数据准备与特征工程
- **清洗**：删除 12 行 `control=True` → 602 行。
- **化学特征**：SMILES 字符 1–3-gram 哈希 → PCA 128 维（本机 rdkit 因 DLL 被系统策略拦截不可用，改用等价字符串核）。
- **`dsig`（药物签名，防泄漏）**：样本 `(c,d)` 的 `dsig` = 药物 `d` 在**除 `c` 之外**的完整细胞类型中的响应均值。
- **`csig`（细胞类型签名）**：`c` 在其 **15 个锚点药**上的响应均值（剔除当前药 `d`）。
- **基线表达（E6）**：把 `adata_train` 中 **control 细胞**按 (cell_type, gene) 聚合成 pseudo-bulk（21,231×6）。
- **紧凑特征** = `[fp_pca(128) | dsig_pca(64) | csig_pca(64) | 2 尺度标量]`；PCA 仅在训练折拟合。

### 1.4 验证协议（官方 test 不可得，自行构造）
- **P1（主，= 社区 AmbrosM 方案）**：目标细胞类型 `c` 只给 **15 个锚点药**；
  训练 = 其余 3 个完整细胞类型全部药 + `c` 的锚点；测试 = `c` 的**非锚点药**（~129，443/129 行）。
- **P2（压力测试）**：`c` 零锚点，测试全部药 → 检验"没有锚点能否迁移"。

### 1.5 泄漏修正（重要）
早期 `build_views` 计算签名时用了**整个 `df`**；因 P1 目标是**完整**细胞类型，其非锚点药标签
（= 测试答案）泄漏进了训练特征。**修正**为签名只读**训练折可见行**（`sanity_checks.py` 用打乱标签负对照验证）。
修正后所有绝对分数显著下降，模型**相对排序**基本不变；本文件以后的结果均基于**修正后**的 pipeline。

---

## 二、模型原理

把问题形式化为**高维输出、极少条件**的结构化预测：输出 18,211 维、预测变量只有 `(细胞类型, 药物)` 两个类别（144×6）。

### 2.1 基线
- `global_mean`（训练列均值）、`drug_mean`（= `dsig`）、`cell_anchor_mean`（= `csig`）、`additive`。

### 2.2 线性 / 核 / 近邻 / 树
- **Ridge / OLS / PLS / KNN**（多输出）；**核岭**（紧凑特征 RBF；高维 `[dsig,csig]` 线性/RBF）。
- **逐基因锚点迁移**：每基因用 15 锚点学 `dsig→y` 的映射。
- **树模型**：因 18k 输出过重，改用**目标空间 PCA 压缩 + 基学习器**（`pca_ridge/xgb/rf/mlp`）。

### 2.3 神经网络与类别 embedding
- **MLP（sklearn/torch）**：紧凑特征 → 多层 → 直接回归 18,211 维（逐基因标准化、Dropout、Adam）。
- **类别 embedding 模型（`EmbMLP`）**：`Emb(cell_id) ⊕ Emb(drug_id) → [Dense→BN→ReLU→Dropout]×2 → Dense(n_dim)`，
  **目标 = TruncatedSVD(n_dim)**，**损失 = row-wise RMSE**（= 官方指标）。这是从公开解法提炼、自行实现的核心模型。
- **混合模型**：embedding + `dsig`(+基线表达) 一起作为输入，兼顾"类别身份"与"跨细胞药物效应"。

### 2.4 低秩目标与降秩回归
- 目标极低秩（见 1.2）。**TruncatedSVD 降维 → 预测成分 → 反变换** 既提速又**去噪**。
- **降秩回归（RRR）**：把系数矩阵约束为低秩，强调"条件共享基因子空间"。

### 2.5 融合与后处理
- **等权/加权集成**：线性/核岭/embedding 误差去相关。
- **预测 clip** 到训练每基因范围；**验证集学习式融合权重**（见 §4.4）。

---

## 三、模型检测

### 3.1 评估指标
- **官方指标 `mrrmse`（mean row-wise RMSE）**：先对每个样本算跨基因 RMSE，再对样本平均（每样本等权）。
- 辅助：MSE / RMSE、R²、per-sample Pearson、Spearman、top-1000 基因 R²。
- **关键发现**：**R² 与 mrrmse 排序不一致**（多次出现）。

### 3.2 正确性 / 无泄漏验证（`sanity_checks.py`，全部 PASS）
- 训练/测试互斥、P1 测试集构造正确；`dsig` 不含目标细胞；
- **打乱标签负对照**：Ridge R²=−0.86（无信号）；完美预测 → R²=1；`drug_mean == dsig`。

### 3.3 主结果（P1，修正后，官方 mrrmse，越小越好）
| 模型 | mrrmse | MSE | R² |
|---|---|---|---|
| **learned-weight 融合** | **0.940** | 1.95 | 0.13 |
| hybrid (ids+dsig) | 0.937–0.948 | 1.94–2.02 | 0.08–0.12 |
| ids+dsig+**baseline** | 0.942 | 1.98 | 0.10 |
| **embedding (秩25, 多种子中位)** | 0.939–0.949 | **1.80** | **0.17** |
| mlp_wide | 1.034 | 2.29 | −0.08 |
| **tuned ensemble** | 1.003 | 2.17 | −0.01 |
| global_mean | ~1.07 | 2.34 | −0.03 |

- **embedding / hybrid 明显优于线性/树/普通 MLP**；`ensemble` 稳健但 mrrmse 稍差。
- **P2（零锚点）所有模型 R²≈0** → **锚点药的少量观测是迁移的必需条件**。

### 3.4 分细胞类型（P1）
- **NK 最易、T CD8+ 最难**（embedding 在 NK R²≈0.5、CD8+ ≈0）。与跨细胞类型响应相关性一致（NK↔CD4+ 最高）。

### 3.5 特征消融
- `dsig` 是**主导特征**（单用 NK R²≈0.42）；`csig` 提供增量；**化学指纹单独几乎无用**；去掉 `dsig` 性能崩塌。

### 3.6 数据驱动方向的检测（多为负结果，附原因）
- 基因子空间投影：对 ensemble 轻微去噪；embedding 无变化。
- 化学相似邻域迁移、逐基因/全局尺度校准、伪标签：**均无效或变差**。
- 单细胞 control 基线表达：**与 dsig 组合有小增益**，单用变差。

---

## 四、参数调试

### 4.1 调参协议
- **外层**：P1 的 4 折（留出目标细胞类型）。
- **内层 / 参考验证**：在外层训练集内再留出一个**参考细胞类型**当目标（同构，带锚点），
  **外层目标非锚点药全程不参与**（`tune_nested.py`）。
- 也曾用"直接用参考细胞选择、在外层目标上观察稳健性"（`tune_refs_cv.py`）。

### 4.2 主要网格与选定参数
| 模型 | 网格 | 选定 |
|---|---|---|
| ensemble | ridge_α{1,10,100}×knn_k{5,10}×pls{10,20} | **ridge_α=100, knn_k=10, pls_n=10** |
| mlp_sklearn | hidden{(256,128),(512,256)}×α{1e-4,1e-3,1e-2} | **(256,128), α=0.01** |
| mlp (torch) | hidden×epoch{100,150}×wd{1e-4,1e-3} | **(512,256), 100ep, wd=1e-3** |
| mlp_wide | (1024,512)×epoch×wd | **(1024,512), 100ep, wd=1e-3** |
| embedding | emb{4,16,64}×SVD秩{8,16,25,35,64} | **emb=64, 秩=25** |
| 种子数 | {1,3,5,7} | **5（中位数）** |

### 4.3 选参可靠性与稳健性（`tune_refs_cv.py`）
- `ensemble`：参考折排名与外层目标排名**完全一致**（12/12）→ 选参可信；
- `mlp`：两者**不一致** → 选参不可靠，稳健融合中剔除；
- `mlp_sklearn` 的 c\* std 最大（0.405）→ 不够稳健。
- **多种子中位数**是最大单项增益（0.9155→**0.9051**，3 折）。

### 4.4 学习式融合权重（`extend_weights.py`）
- 在参考折上以 mrrmse 拟合凸组合（softmax+Nelder–Mead），分量 `{embed, hybrid, ensemble, compound, cell}`。
- 结果：**embed 0.15 / hybrid 0.80 / ensemble 0.05，compound 0，cell 0**；
  自动把冗余/噪声分量压到 0（与消融结论一致）。外层 mrrmse **0.940**（优于固定权重 0.945）。

---

## 五、最终结论

### 5.1 有效 vs 无效
**有效**
1. **低秩输出**（SVD/PCA 秩≈25）：提速 + 去噪，反复被验证。
2. **类别 embedding + row-wise RMSE 损失**：最强单模型，优于线性/树/普通 MLP。
3. **多种子中位数**：最大单项方差缩减增益，简单稳健。
4. **`dsig` 跨细胞药物签名**：核心可迁移特征。
5. **与 `ensemble` 融合 / 参考折学习式权重**：小幅但稳定提升。
6. **单细胞 control 基线表达**：与 `dsig` 组合有小增益。

**无效（含原因）**
- 朴素迁移（anchor-mean/additive）、残差迁移、逐基因迁移 → 过拟合/尺度错配。
- 化学相似邻域迁移、逐基因/全局尺度校准、伪标签 → 噪声大或循环强化。
- 聚合先验融合、target-encoded Ridge+TSVD → 与 `dsig` 重复或缺乏正则。
- **P2（零锚点）**：全部失效 → 目标细胞的少量观测不可替代。

### 5.2 反复出现的现象：指标冲突
**R² / MSE 与官方 mrrmse 排序不一致**：mrrmse 每样本等权、强调整体尺度匹配；
MSE 被极少数大幅值基因主导。因此本项目的选参/选模**以 mrrmse 为准**（embedding/hybrid 更好），
同时报告 MSE/R² 以暴露权衡。

### 5.3 最终模型与提交
- **最终主提交 `results/submission.csv`**
  = `0.15 · embedding(ids, SVD秩25) + 0.80 · hybrid(ids + dsig + baseline) + 0.05 · ensemble`，
  NN 取多种子中位，权重来自参考折拟合；格式核验通过（255×18212、列/序一致、无 NaN）。
- **P1 官方 mrrmse 提升轨迹**：`ensemble` 1.003 → `embedding` 0.949 → `hybrid+baseline` 0.942 → **融合 0.940**。

### 5.4 局限与后续
- 目标细胞仅 15 行标签，离线 4 折代理与真实 B/Myeloid 未必完全一致（离线 R² 排名曾与榜上不一致）。
- 单细胞/multiome 仅用到 control pseudo-bulk；可进一步用全细胞表达、细胞相似度先验、层次/张量分解。
- 环境提示：会话中段本机 torch 单次拟合约 11s→85s，长配置脚本多次超时，上表以 1–2 种子、4 折完成。

### 5.5 复现命令（简）
```powershell
cd G:\大学\大三上\统计学习\project\exp
python sanity_checks.py                 # 无泄漏/正确性
python tune_nested.py --model <m>       # 嵌套/参考调参
python extend_research.py               # 低秩/种子/伪标签
python compute_baseline.py              # control pseudo-bulk
python extend_baseline2.py              # 基线特征
python extend_weights.py                # 学习式融合权重
python submit_final.py                  # 生成 submission.csv
```

### 5.6 关键目录
```
exp/
├─ common.py models.py run_benchmark.py analyze.py        # 主框架
├─ tune_nested.py tune_refs_cv.py                          # 调参
├─ improve_embed.py improve_embed2.py improve_rank.py      # embedding 模型
├─ extend_research.py extend_data.py extend_hybrid.py
├─ extend_target.py extend_baseline2.py extend_weights.py
├─ compute_baseline.py submit*.py
└─ results/  (指标 csv、predictions、figures/、REPORT.md)
```

---

## 六、第二轮：新模型、损失与调参（详见 `exp/REPORT_ROUND2.md`）

在固定（无泄漏）P1、官方 mrrmse 下，本轮做了四件事并写入新报告 `exp/REPORT_ROUND2.md`：

1. **补学公开解法**：目标自编码器（JAX）、TabTransformer/SAINT、SMILES 字符 CNN（NLP）、
   PyBoost 多输出 GBDT、RegressorChain —— 逐条说明“在做什么 + 是否采用 + 原因”。
2. **新模型**：目标自编码器（非线性低秩，❌ 0.963）、CP 张量分解（❌ 1.275）、
   SMILES 字符 CNN（✅ 0.904，与纯 id embedding 打平）。
3. **损失函数**：逐样本 RMSE 0.910 最优 > Huber/log-cosh > MSE 0.969。
4. **系统调参**：emb 64→128、网络加深到 (256,512,1024)、lr 1e-3→3e-3、dropout 0.2，
   把单模型 mrrmse 从 **0.944 降到 0.899**（5 种子中位），优于旧报告的任何集成（0.940）。

**P1 mrrmse 轨迹**：`ensemble 1.003 → embed(秩25) 0.944 → 调参后 0.900 → +5 种子中位 0.899`。
新增代码 `exp/round2_common.py`、`exp_ae.py`、`exp_bilinear.py`、`exp_smiles.py`、`exp_tune.py`、
`exp_final.py`、`submit_round2.py`、`robust_lib.py`、`exp_robust.py`、`submit_robust2.py`；
结果 `exp/results/round2_*_{raw,mean}.csv`。

5. **稳健性**：模仿第一名的「随机 K 折 bagging」在本数据上**更差**（训练集仅 ~440–590 行，
   下采样丢信息 > 降方差）；稳健做法是「全量数据 × 多种子 × 多模型族 + 固定权重」。

**提交**：
- **主提交 `exp/results/submission.csv` = 稳健集成**（`submission_robust2.csv`）：
  `0.50·embed(7 种子) + 0.30·smiles(3 种子) + 0.20·hybrid(3 种子)`，全量训练、固定权重。
  旧提交备份 `submission_prev2.csv`。
- 备选 `exp/results/submission_round2.csv` = 调优单模型（5 种子中位，P1 mrrmse 0.899），
  旧提交备份 `submission_prev.csv`。

格式均核验通过（255×18212、列/序一致、无 NaN）。
