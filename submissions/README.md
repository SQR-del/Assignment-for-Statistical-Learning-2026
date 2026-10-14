# submissions：已获得官方分数的提交文件

> **本地副本说明**：这份说明复制自服务器。服务器上的 `methods_20261013/`，在本地对应 `methods/upload_7z/`。本文件夹里还有早期行顺序错误的文件（`krr_null_v1.csv` 等），它们的分数无效。

所有文件都按 Kaggle id 顺序排列：按 (cell_type, sm_name) 排序，0–127 为 B 细胞，128–254 为 Myeloid。

| 文件 | 模型 | private | public | sha256 |
|---|---|---:|---:|---|
| **`blend_v1_kaggle_ids.csv`** | **最终模型**：0.4·(底座 + I2) + 0.4·(底座 + I1 + I2) + 0.2·KRR v2 | **0.752** | **0.591** | 16a8ebaf… |
| `nn_base_I2_v1_kaggle_ids.csv` | 底座 + I2 幅度输出头 | 0.753 | 0.597 | 0b78abc2… |
| `nn_base_v1_posthoc_kaggle_ids.csv` | 底座（事后消融） | 0.813 | 0.595 | e03f8504… |
| `krr_null_v1_kaggle_ids.csv` | KRR v1（null 参照 + 锚点类比 KRR） | 0.830 | 0.604 | 8e2a0d2c… |

- 参考：private 第 1 名 0.729，金牌线 ≤ 0.745，银牌线 ≤ 0.758，中位数 0.793。
- 这些文件都是赛后提交的，模型选择没有使用排行榜。
- 用 `code/predict.py` 可以重新生成最终文件，见 `results/blend_final/`。
- 早期三个行顺序错误的提交已移到 `../archive/`，其分数无效：krr_null 0.981、L3 1.201、baseline 1.558。

## 2026-10-13 补充提交（methods/ 中尚无官方分数的方法）

文件在 `methods_20261013/`，就是实际上传的 7z 压缩包。为满足上传大小限制，00–03 保留 4 位小数，04、07、09 保留 3 位小数，对分数的影响小于 0.0003。全精度版本在 `results/methods/`。这些提交只用于评估已经固定的方法，不用于选模型。

| 文件 | 方法 | private | public |
|---|---|---:|---:|
| 00_zero_kaggle_ids.7z | 全零 | 0.902 | 0.666 |
| 01_target_mean_kaggle_ids.7z | 目标细胞均值 | 1.496 | 1.000 |
| 02_null_median_kaggle_ids.7z | 目标细胞中位数（null 参照） | 0.896 | 0.652 |
| 03_compound_mean_kaggle_ids.7z | 同化合物在其他细胞中的均值 | 0.839 | 0.627 |
| 04_null_plus_shift_kaggle_ids.7z | null 参照 + 来源偏移 | 0.848 | 0.631 |
| 07_nn_base_I1_kaggle_ids.7z | 底座 + I1 | 0.798 | 0.588 |
| 09_nn_base_I1_I2_kaggle_ids.7z | 底座 + I1 + I2 | 0.764 | 0.614 |
| 11_kernel_te_kaggle_ids.7z | 联合 KRR（即 methods/05b_krr_joint，提交时用的旧名；2026-10-14 提交） | 0.823 | 0.598 |
| 05a_krr_local_kaggle_ids.7z | 局部 KRR（KRR v2 配置，2026-10-14 提交） | 0.799 | 0.594 |

另外，重排后的 LGT L3（archive/results_lgt/kaggle_id_fix/L3_kaggle_ids.csv）在 2026-09-30 已提交，官方分数为 0.901 / 0.669。
