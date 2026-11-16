# repro/ — 已提交模型的复现代码

本目录把此前散落在 `exp/submit*.py`、`exp/apply_tuned.py`、`exp/submit_round2.py`
里的“生成 Kaggle submission”逻辑重新整理成**独立、可重复运行**的脚本。
每个脚本只依赖根目录的 `code/`（`common.py` / `models.py` / `nn.py`）和
`data/` 下的三个文件：

```
data/de_train.parquet
data/id_map.csv
data/sample_submission.csv
```

> 复现 B / Myeloid 的提交不需要 `adata_train.parquet` 等单细胞大文件。

---

## 1. 六个提交与脚本对照

| # | 模型 | 归档提交（本仓库 `results/`） | 复现脚本 | 关键配置 | 与归档 CSV 的对齐度 |
|---|---|---|---|---|---|
| 16 | tuned embedding | `submission_round2.csv` | `16_embedding.py` | emb=128, hidden=(256,512,1024), SVD=25, 400ep, lr=3e-3, wd=1e-4, drop=0.2, **5 seeds median** | 1 seed: corr 0.967；5 seeds 为原配方 |
| 20 | PCA + Ridge | `submission_pca_ridge.csv` | `20_pca_ridge.py` | target PCA 128, Ridge α=10, 199 维 compact 特征 | **EXACT**（corr 1.000000, rowRMSE 0.0000） |
| 21 | sklearn MLP | `submission_mlp_sklearn.csv` | `21_mlp_sklearn.py` | hidden=(256,128), α=0.01, early stopping | **NEAR**（corr 0.999360, rowRMSE 0.0413） |
| 22 | wide torch MLP | `submission_mlp_wide.csv` | `22_mlp_torch_wide.py` | hidden=(1024,512), **150ep, wd=1e-4**, dropout=0.1, cosine | **NEAR**（corr 0.994524, rowRMSE 0.1342） |
| 23 | classic ensemble | `submission_ensemble.csv` | `23_ensemble.py` | target-PCA Ridge(α=100) + Ridge(100) + PLS(10) + KNN(10) + linear KRR | **CLOSE**（corr 0.982024, rowRMSE 0.2111） |
| 24 | 第一轮融合 | 旧 `submission.csv`（0.902/0.659） | `24_first_round_blend.py` | **0.6·embed(emb=64, rank25, 5 seeds) + 0.4·ensemble** | 1 seed + improve 配方 corr 0.981；原配方需 5 seeds |

对齐度用 `repro/check_submissions.py` 计算：把复现 CSV 与归档 CSV 按 `id` 对齐后，
逐元素求相关系数、逐行 RMSE 和逐行相关中位数。Kaggle 的 private 标签不公开，
所以**归档 CSV 是“同一份预测”的代理标准**，而不是用 LB 分数反推。

`results/` 里同时保留了当时的 Kaggle 分数（私有 / 公开）：

| # | 模型 | 我们生成 | 组员原提交 | 私有榜差 | 结论 |
|---|---|---|---|---|---|
| 16 | embedding | 0.772 / 0.573 | 0.762 / 0.568 | +0.010 | 对得上 |
| 20 | PCA + Ridge | 1.031 / 0.728 | 1.028 / 0.732 | +0.003 | 对得上 |
| 23 | classic ensemble | 0.878 / 0.637 | 0.885 / 0.645 | −0.007 | 对得上 |
| 24 | 第一轮融合 | 0.898 / 0.653 | 0.902 / 0.659 | −0.004 | 对得上 |
| 21 | sklearn MLP | 0.862 / 0.636 | 0.893 / 0.659 | −0.031 | 对不上（旧代码更好） |
| 22 | wide MLP | 1.066 / 0.752 | 0.954 / 0.686 | +0.112 | 对不上（旧代码差很多） |

---

## 2. 复现时“对不上”的原因（已定位）

### 2.1 经典模型的特征维度错了（最关键）

`code/predict.py::build_target_context` 是第二轮融合时写的简化版本：

```python
compact = [fp_pca(128) | dsig_pca(64)]          # 192 维
```

但验证协议 `common.build_views` 实际使用的是：

```python
compact = [fp_pca(128) | dsig_pca(64) | csig_pca(5) | 2 个 scale]   # 199 维
```

也就是说，导出提交时**漏掉了 cell signature（csig）和两个尺度特征**，
并且把 `Csig_train/test` 直接设成了 0。经典模型（`pca_ridge`、`ensemble`、
`mlp_sklearn`、`mlp_torch`）在验证时用 199 维训练，导出时却用 192 维，
自然对不上。

`repro/target_context.py` 重新实现了与 `common.build_views` **完全一致**的
目标细胞（B / Myeloid）特征构造，并且用归档文件验证过：
`pca_ridge` 的复现结果与 `submission_pca_ridge.csv` 逐元素相关系数
= 1.000000，逐行 RMSE = 0.0000。

### 2.2 wide MLP 用了“调参后”的配置

`submission_mlp_wide.csv` 是**早期** `submit.py` 生成的：

```
hidden=(1024,512), 150 epochs, lr=1e-3, wd=1e-4, dropout=0.1, cosine
```

后来的嵌套 CV 把 wide MLP 调成了 `100 epochs, wd=1e-3`，并写进了
`methods/06_mlp_torch` 附近的文档。这个“调参后”的配置在真实 B/Myeloid 上更差，
正是私有榜 +0.112 的主要来源。`22_mlp_torch_wide.py` 默认回到 **150ep / wd=1e-4**，
`--tuned` 才使用 100ep / wd=1e-3。

### 2.3 sklearn MLP 的差异主要来自 2.1

`mlp_sklearn` 配置本身是对的（hidden=(256,128), α=0.01）。换成正确的 199 维特征后，
复现结果与归档 CSV 的 corr = 0.999360，逐行 RMSE = 0.0413；剩余差异来自
sklearn 版本的随机初始化 / 早停细节。

### 2.4 第一轮融合不是第二轮融合

第二轮主提交 `code/predict.py` 是：

```
0.50·embedding + 0.30·SMILES-CNN + 0.20·hybrid
```

而“第一轮融合”是：

```
0.60·embedding(emb=64, SVD=25, 5 seeds median) + 0.40·ensemble
```

两者不要混用。`24_first_round_blend.py` 默认生成第一轮配方；
`--recipe improve` 生成接近 `submission_improve_blend.csv` 的 0.65/0.35 版本。

### 2.5 embedding 是 5-seed 中位数，单 seed 会偏

`submission_round2.csv` 是 5 个随机种子的**中位数**。单 seed 复现的 corr 约 0.967，
且 Myeloid 的预测幅度会明显偏大（1 seed mean|p|≈0.90，5 seeds 归档≈0.58）。
所以不要用 1 seed 的结果去和归档比幅度；默认 `--reps 5`。

### 2.6 ensemble 只能做到 CLOSE

归档 `submission_ensemble.csv` 与当前 `m_ensemble` 的 corr = 0.982、rowRMSE = 0.211。
差异很可能来自早期版本的 `ensemble` 组件（例如 `krr_rbf_sig` 与 `krr_linear_sig` 的取舍、
早期特征版本或随机种子），报告里也记录了 ensemble 在 §8 与 §14 用过两种组合。
当前脚本按报告 §14 的最终配置实现；如果需要**逐位一致**，请直接使用归档 CSV。

---

## 3. 运行方式

先安装依赖并准备数据（见根 `README.md`）：

```bash
pip install -r requirements.txt
```

然后从仓库根目录运行：

```bash
# #20 PCA + Ridge（最快，几秒）
python repro/20_pca_ridge.py --out results/repro/submission_pca_ridge.csv

# #21 sklearn MLP（约 1–2 分钟）
python repro/21_mlp_sklearn.py --out results/repro/submission_mlp_sklearn.csv

# #22 wide MLP（CPU 约 2–3 分钟；默认 150ep / wd=1e-4）
python repro/22_mlp_torch_wide.py --out results/repro/submission_mlp_wide.csv

# #23 classic ensemble（约 20 秒）
python repro/23_ensemble.py --out results/repro/submission_ensemble.csv

# #16 tuned embedding（5 seeds，CPU 较慢；可先用 --reps 1 检查流程）
python repro/16_embedding.py --reps 5 --out results/repro/submission_round2.csv

# #24 第一轮融合（5 seeds；--recipe improve 为 0.65/0.35）
python repro/24_first_round_blend.py --recipe first --embed-reps 5 \
    --out results/repro/submission_first_round_blend.csv
```

查看对齐度：

```bash
python repro/check_submissions.py --all --by-cell
python repro/check_submissions.py \
    results/repro/submission_pca_ridge.csv results/submission_pca_ridge.csv --by-cell
```

`--all` 会检查 `results/repro/` 下已存在的文件；缺失的文件会跳过。
`EXACT` 表示 corr > 0.999 且 rowRMSE < 1e-3，`NEAR` 表示 corr > 0.99，
`CLOSE` 表示 corr > 0.95。

### 特征一致性自检（可选，较慢）

```bash
python repro/target_context.py
```

它会用 `common.build_views` 在 B cells 上重建一次参考特征，并和
`repro/target_context.py` 的快速实现逐项比较（`Xc_train/Xc_test/Dsig/Csig`）。
预期所有项 `allclose = true`。

---

## 4. 旧代码 → 新代码对照

`exp/` 目录在项目重组时被合并/删除，旧脚本不在当前仓库里；下面的对应关系来自
`REPORT.md`、`REPORT_ROUND2.md` 和归档文件名：

| 归档提交 | 旧生成脚本（据报告） | 报告位置 | 新脚本 |
|---|---|---|---|
| `submission_round2.csv` | `submit_round2.py` | `REPORT_ROUND2.md` §3.6/§3.9 | `16_embedding.py` |
| `submission_pca_ridge.csv` | `submit.py` / `apply_tuned.py` | `REPORT.md` §12/§13.5 | `20_pca_ridge.py` |
| `submission_mlp_sklearn.csv` | `apply_tuned.py` / `submit.py` | `REPORT.md` §13.5/§12 | `21_mlp_sklearn.py` |
| `submission_mlp_wide.csv` | `submit.py`（早期 `mlp_torch_wide`） | `REPORT.md` §8/§12 | `22_mlp_torch_wide.py` |
| `submission_ensemble.csv` | `apply_tuned.py` / `select_robust.py` | `REPORT.md` §13.5/§14.3 | `23_ensemble.py` |
| 旧 `submission.csv`（0.902/0.659） | `submit_improve.py` / `submit.py` | `REPORT.md` §16.3/§17.7 | `24_first_round_blend.py --recipe first` |
| `submission_improve_blend.csv` | `submit_improve.py` | `REPORT.md` §15.3/§16.3 | `24_first_round_blend.py --recipe improve` |

> 第二轮主提交 `submission.csv`（`0.50·embed + 0.30·SMILES + 0.20·hybrid`）
> 仍由 `code/predict.py` 生成，不在本目录的六个“历史提交”范围内。

---

## 5. 已验证结果

使用本目录脚本在本机（Windows / CPU / torch 2.6.0+cpu / sklearn 1.9.0）生成的
`results/repro/*` 与归档 `results/submission_*.csv` 对比：

| 模型 | 状态 | corr | mean row RMSE | mean\|new\| | mean\|ref\| |
|---|---|---:|---:|---:|---:|
| #20 pca_ridge | EXACT | 1.000000 | 0.0000 | 1.036 | 1.036 |
| #21 mlp_sklearn | NEAR | 0.999360 | 0.0413 | 0.818 | 0.820 |
| #22 mlp_wide（150ep/wd1e-4） | NEAR | 0.994524 | 0.1342 | 0.902 | 0.908 |
| #23 ensemble | CLOSE | 0.982024 | 0.2111 | 0.640 | 0.737 |
| #16 embedding（1 seed） | CLOSE | 0.967071 | 0.1913 | 0.394 | 0.422 |
| #24 improve blend（1 seed） | CLOSE | 0.980791 | 0.2445 | 0.648 | 0.526 |

其中 #16 / #24 的归档是 5-seed 中位数；本机跑满 5 seeds 约需数十分钟到两小时
（取决于 CPU），所以表中用 1 seed 验证流程。跑满后 corr 会进一步提高，幅度也会回到归档水平。

---

## 6. 常见问题

**Q: 为什么 `methods/harness.py --export` 生成的经典模型对不上？**
A: 它调用 `code/predict.py::build_target_context`，继承了 192 维的旧特征。
请用本目录的脚本；或者先把 `build_target_context` 改成与 `common.build_views` 一致。

**Q: 我的 wide MLP 分数很差？**
A: 检查是不是用了 `100ep / wd=1e-3`。归档提交是 `150ep / wd=1e-4`；
`22_mlp_torch_wide.py` 默认就是后者，`--tuned` 才是前者。

**Q: 为什么 embedding / 第一轮融合的预测幅度比归档大？**
A: 归档是 5-seed 中位数，单 seed 的 Myeloid 预测会偏大；请用 `--reps 5` / `--embed-reps 5`。

**Q: 可以逐位复现所有归档 CSV 吗？**
A: `pca_ridge` 可以（EXACT）。其余模型含随机初始化或早期代码细节，
在相同数据和配置下是 NEAR/CLOSE；如果需要逐位一致，直接使用归档 CSV。
