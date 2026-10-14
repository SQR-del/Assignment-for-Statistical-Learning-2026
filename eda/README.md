# eda/ 探索性数据分析

数据来源与形状等关键事实（见根 `README.md` / `SUMMARY.md` §一）。

| 文件 | 内容 |
|---|---|
| `01_explore.py` | 生成下面 5 张图；`python eda/01_explore.py`（依赖 `code/common.py` 的数据路径） |
| `01_label_distribution.png` | 标签量级与有符号分布（长尾，81.5% 的 \|值\|<1，最大 ≈180） |
| `02_coverage.png` | 每个细胞类型观测到的药物数（4 个完整细胞 144 药；B/Myeloid 仅 15 个锚点药） |
| `03_strength.png` | 各细胞类型每药响应强度（B/Myeloid 明显更强） |
| `04_crosscell_corr.png` | 跨细胞类型同药响应相关性（NK↔CD4+ 最高，是 P1 迁移的根据） |
| `05_top_genes.png` | 全局最“响应”的基因 |

**关键发现**：目标矩阵极低秩（第 1 主成分解释 65.5%，90% 只需 23 个）；缺的是**新细胞类型**
而不是新药物（129 个测试药在别的细胞类型里都被观测过）。这些事实支撑了 P1 协议与 dsig 特征。
