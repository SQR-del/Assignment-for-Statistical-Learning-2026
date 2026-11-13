# 统计学习大作业：单细胞扰动预测（NeurIPS 2023 Kaggle）

这里是本地副本：代码、各方法的说明和分数、各次实验的汇总结果。完整项目（数据、全部预测文件、中间结果）在服务器 pku-server 上：

```text
/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/
```

代码需要服务器上的数据和 GPU 才能运行，本地只用于阅读和写报告。

## 最终结果

blend = 0.4 · (底座 + I2) + 0.4 · (底座 + I1 + I2) + 0.2 · 局部 KRR（KRR v2 配置）

- 官方分数：private 0.752 / public 0.591，在 1098 支队伍中约排第 23 名。
- 内部验证的 objective 为 0.927。

2026-11-13 更新：同一配方、网络改用 nn_v2 的设置（K=32，800 轮，dropout 0.2）后，内部 objective 0.920，官方 private 0.734 / public 0.573（赛后提交，设置在提交前已由内部验证确定）。见 `methods/README.md` 末尾和 `methods/nn_v2_lab/README.md`。

## 本地目录

| 目录 | 内容 | 从哪看起 |
|---|---|---|
| `code/` | 最终模型代码：`predict.py` 生成提交文件，`validate.py` 跑内部验证 | `code/README.md` |
| `methods/` | 12 个基线和代表性方法，每个一个文件夹，含做法、运行脚本、内部分数和官方分数 | `methods/README.md`（总表） |
| `results/` | 各次结果的汇总：`validation/`（内部验证）、`blend_final/`（最终预测的检查记录）、`nn_v2_lab/`（nn_v2 的全部实验）以及历次实验的脚本和分数 | `results/README.md` |
| `submissions/` | 提交文件。`methods/` 子文件夹有 12 个方法的全精度 CSV，以及实际上传的 `upload_7z/` | `submissions/README.md`、`submissions/methods/README.md` |
| `Question 2/` | 标签生成研究：按主办方原始脚本重建标签（与官方逐数值一致）、小模型 / 大模型模拟、噪声来源分析。脚本、说明和汇总结果 | `Question 2/README.md` |
| `archive/` | 已移除或归档的方法（LGT、底座 + 收缩后处理、KRR v1）的说明和分数 | `archive/README.md` |
| `开题报告/` | 开题报告（未改动） | — |

## 本地没有的内容（只在服务器上）
- `data/`：原始数据，约 3.8 GB。
- `Question 2/` 下的数据缓存和拟合结果（`cache_official/`、`official_fit/`、`sim/`、`weights_exp/`，约 4 GB）以及 `adata_excluded_ids.csv.zip`。
- `results/methods/`：各方法的全精度提交文件。本地的同样文件在 `submissions/methods/`。
- `results/` 下其余的 `*_kaggle_ids.csv` 和 `*.pkl`。
- `archive/` 中的旧 LGT 代码与结果。
