# methods_removed：已从 methods/ 移除的方法（2026-10-13）

移除规则：比某个保留的、更简单的方法复杂，却在 A15、V1w、V1 三项上全都更差，且不是最终模型的组成部分。

| 文件夹 | 方法 | 被谁全面超过 | objective |
|---|---|---|---:|
| 05_lgt | 原 LGT（L3），调用 archive/code_lgt_20260930 | methods/03_compound_mean | 1.116 |
| 09_nn_base_shrink | 底座 + 第 4 名收缩后处理 | methods/06_nn_base | 0.982 |
| 05a_krr_v1 | KRR v1（局部 KRR 第一版；methods/05a_krr_local 改用 v2 配置后归档，并非被全面超过；仍是 objective 的基准，官方 0.830 / 0.604） | — | 1.000 |

各文件夹的 README、scores.json 和 rows.csv 都保留着，run.py 仍可运行（它会使用 methods/harness.py）。
