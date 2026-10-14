# results 目录索引

**最终结果**：`blend_final/` 和 `validation/`，都由 `code/` 里的当前代码生成。**Kaggle 官方分数**：`kaggle_scores.csv`，说明见下文。

## Kaggle 官方分数

### 记录在哪里
- **`kaggle_scores.csv`**：本项目在 Kaggle 上的全部 17 次提交，每次一行。

  | 列 | 含义 |
  |---|---|
  | `kaggle_file` | Kaggle 提交记录页上显示的文件名 |
  | `submitted` | 提交日期 |
  | `method_folder` / `method` | 对应的方法（`methods/` 下的文件夹名）和中文名 |
  | `private` / `public` | 官方分数（MRRMSE，越低越好） |
  | `valid` | 分数是否有效 |
  | `precision` | 上传文件的数值精度 |
  | `uploaded_file` / `uploaded_sha256` | 实际上传的文件和它的 sha256 |
  | `full_precision_file` | 同一方法用当前代码导出的全精度文件 |
  | `note` | 备注，例如与当前导出文件是否相同 |

- **`methods/<方法>/kaggle_score.json`**：每个方法自己的有效官方分数，以及对应的上传文件。

### 怎么理解这些分数
- **来源**：比赛已经结束，所有提交都是赛后提交（Late Submission）。赛后提交不进入排行榜，但会给出 private 和 public 两个分数；分数是从比赛的 Submissions 页面读取的。
- **private 与 public**：测试集共 255 行（129 个化合物 × B/Myeloid 两种细胞），分成 public（约 50 个化合物）和 private（约 79 个化合物）两部分。比赛最终排名看 private，所以我们也以 private 为主。
- **有效与无效**：最早的 3 次提交行顺序错误，用的是 `data/id_map.csv` 的交错顺序，而不是 Kaggle 要求的 (cell_type, sm_name) 排序，所以分数无效，只作记录。
- **数值精度**：2026-10-13 起的提交，受浏览器上传工具 10 MB 的限制，压缩成 7z 上传，数值保留 4 位或 3 位小数，对分数的影响小于 0.0003。全精度版本在 `methods/<方法>/`。
- **用途**：这些分数只用于事后评估已经固定的方法。最终模型 blend 是按内部验证预先选定的，没有用排行榜调参或选模型。
- **参照**：private 第 1 名 0.729，金牌线 ≤ 0.745，银牌线 ≤ 0.758，所有队伍的中位数 0.793。

### 各方法的有效官方分数
| 方法（`methods/` 文件夹） | private | public | 提交日期 |
|---|---:|---:|---|
| 00 全零 | 0.902 | 0.666 | 2026-10-13 |
| 01 目标细胞均值 | 1.496 | 1.000 | 2026-10-13 |
| 02 目标细胞中位数（null 参照） | 0.896 | 0.652 | 2026-10-13 |
| 03 同化合物在其他细胞中的均值 | 0.839 | 0.627 | 2026-10-13 |
| 04 null 参照 + 来源偏移 | 0.848 | 0.631 | 2026-10-13 |
| 05a 局部 KRR（KRR v2 配置） | 0.799 | 0.594 | 2026-10-14 |
| 05b 联合 KRR | 0.823 | 0.598 | 2026-10-14 |
| 06 神经网络底座 | 0.813 | 0.595 | 2026-09-30 |
| 07 底座 + I1 | 0.798 | **0.588** | 2026-10-13 |
| 08 底座 + I2 | 0.753 | 0.597 | 2026-09-30 |
| 09 底座 + I1 + I2 | 0.764 | 0.614 | 2026-10-13 |
| **10 blend（最终）** | **0.752** | 0.591 | 2026-09-30 |
| KRR v1（已归档，见 `archive/methods_removed/05a_krr_v1`） | 0.830 | 0.604 | 2026-09-30 |
| LGT（已移除，见 `archive/methods_removed/05_lgt`） | 0.901 | 0.669 | 2026-09-30 |

## 目录
| 目录或文件 | 内容 | 官方分数（private / public） |
|---|---|---|
| `kaggle_scores.csv` | 全部 Kaggle 提交及分数的总记录，见上文 | — |
| `blend_final/` | 最终模型，由 `code/predict.py` 生成：`blend_kaggle_ids.csv` 及三个组成部分；`summary.json` 记录泄漏检查、sha256 和与先前导出文件的差异 | 与 blend_v1 相同（最大差异 8.8e-8） |
| `validation/` | 内部验证，由 `code/validate.py` 生成：`rows.csv` 为逐行误差，`summary.json` 为汇总和 bootstrap 置信区间 | — |
| `blend_v1/` | 实际提交的 blend 文件（sha256 16a8ebaf…）和当时的组合脚本 | **0.752 / 0.591** |
| `nn_base_I2_v1/` | 实际提交的"底座 + I2"，以及当时的消融脚本和全部消融运行记录（`ablation_runs/`） | 0.753 / 0.597 |
| `nn_base_I1_I2_v1/` | "底座 + I1 + I2"，blend 的组成部分 | 0.764 / 0.614（2026-10-13 以 `09_nn_base_I1_I2` 的名义提交，3 位小数） |
| `nn_base_v1_posthoc/` | 底座，事后消融用 | 0.813 / 0.595 |
| `krr_v2/` | KRR v2，即局部 KRR（`methods/05a_krr_local`），blend 的组成部分 | 0.799 / 0.594（2026-10-14 以 `05a_krr_local` 的名义提交，3 位小数） |
| `krr_null_v1/` | KRR v1（局部 KRR 第一版，已归档）：超参数选择记录、内部分数；`krr_null_v1_kaggle_ids.csv` 是提交文件 | 0.830 / 0.604 |
| `methods/` | 12 个方法的全精度提交文件（由 `methods/<方法>/run.py --export` 生成），以及各自的 `kaggle_score.json` | 见上表 |
| `krr_joint_study/` | 联合 KRR 的参数网格（140 个配置），用来区分数据和学习器的作用；见 `methods/05b_krr_joint` | — |
| `label_generation_study/` | 标签生成研究：复现 limma 流程，做少样本与全样本拟合的模拟和常数基因检验，建立对齐标签生成方式的验证，并评估校正效果（结论：不采用） | — |
| `pseudobulk_train/` | 由训练部分原始计数生成的伪 bulk 矩阵和各版本的 limma 结果，供标签生成研究使用（约 2 GB） | — |
| `../methods/`（项目根目录） | 基线和代表性方法，每个方法一个文件夹，含做法说明、运行脚本和内部验证分数 | 见 `methods/README.md` |

以下内容已移到 `../archive/`：
- 原 LGT 模型的结果：`lgt_20260930_v1`。
- 重排 id 用的临时目录：`kaggle_id_fix`，里面有重排后的 L3 和 baseline。
- 早期行顺序错误的 `krr_null.csv`。
