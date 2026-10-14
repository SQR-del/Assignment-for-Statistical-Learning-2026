# 11_bilinear_cp 双线性 / CP 张量分解

**做法**：直接建模 (cell×drug×gene) 响应张量 Y[c,d,g] ≈ μ[g]+A[c,g]+B[d,g]+Σ_k U[c,k]W[d,k]Z[k,g]，用掩码 Adam 拟合观测项；新细胞类型只在15 个锚点上估计 A、U。**结论：明显更差**（尺度错配）。

**关键参数**：`R`（秩）、`steps`、`lr`、`wd`。
