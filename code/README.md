# code/ 核心库

所有模型都做了封装：以函数形式提供，**必要超参作为关键字参数**，便于直接调用与调参。

| 文件 | 内容 |
|---|---|
| `common.py` | 数据加载（`load_de`）、锚点药（`anchor_drugs`，默认排除 2 个对照）、索引（`cat_indices`）、指纹 PCA（`drug_fingerprint_pca`）、折构造（`make_folds` / `inner_folds`）、泄漏安全的 dsig/csig 特征（`build_views`，可用 `source_cells=` 指定药物签名来源细胞）、指标（`compute_metrics`，`mrrmse`、`rms`）。数据路径可用环境变量 `SCP_DATA_DIR` 覆盖。 |
| `teammate_metric.py` | **默认模型评估口径**（与队友 `main` 分支一致）：34 折 A + V1，A15 / V1w / `objective = mean(V1w/1.1625, A15/0.9894)`；含快速特征构造与逐折 `rows.csv`。 |
| `models.py` | 经典（numpy / sklearn）模型族：基线、线性/PLS/KNN、核方法、逐基因锚点迁移、树、目标空间 PCA、sklearn-MLP、降秩回归 RRR、稳健集成。统一签名 `fn(view, seed=..., **params) -> (n_test, G)`。注册表 `CLASSIC_METHODS`。 |
| `nn.py` | torch 模型族：全输出 MLP、类别 embedding MLP（`train_embed`）、混合模型（`train_hybrid`）、SMILES 字符 CNN（`train_smiles`）、目标自编码器（`train_ae_target`）、双线性/CP（`fit_cp`/`predict_cp`）。损失与低秩压缩（`make_loss`、`Reducer`）也在其中。 |
| `blend.py` | 融合：`learn_simplex_weights`（在验证折上按官方 mrrmse 学权重）、`blend`、`fixed_blend`。 |
| `validate.py` | 基准驱动 + 自检：`python code/validate.py` 默认跑队友指标（A15/V1w objective），`--metric p1p2` 跑旧 P1/P2；`--sanity` 做无泄漏检查。 |
| `predict.py` | 最终提交：`python code/predict.py`，默认 `0.50·embedding + 0.30·SMILES-CNN + 0.20·hybrid`（多随机种子）。 |

## 直接调用示例

```python
import sys; sys.path.insert(0, "code")
import common as C, models as M, nn

df, genes = C.load_de()
anchors = C.anchor_drugs(df)
fp = C.drug_fingerprint_pca(df, 128)
fold = C.make_folds(df, genes, anchors, "P1")[0]
view = C.build_views(df, genes, fold, fp)

pred = M.m_ridge(view, alpha=100)                 # 调 alpha
pred = M.m_ensemble(view, ridge_alpha=100, knn_k=10, pls_n=10)
pred = M.m_rrr(view, rank=20, alpha=10)
```

## 依赖

见项目根目录 `requirements.txt`。数据放置见根 `README.md`。
