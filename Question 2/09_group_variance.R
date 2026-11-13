# 09_group_variance.R  各组样本的残差方差（相对 voom 的预测），6 种细胞类型的训练拟合
# 用法: Rscript 09_group_variance.R <核数>      输出: out/group_variance.csv
# 对每个基因 g、每组样本 G（DMSO / 阳性对照 PC / 普通药 ORD）:
#     v_G(g) = sum_{i in G} w_gi * r_gi^2 / sum_{i in G} (1 - h_gi)
# r 为加权拟合的残差，h 为该基因加权设计矩阵的杠杆值。若 voom 的方差模型对每组都成立，三组的 v 应相同。
# 另外按"小模型 / 大模型"的残差自由度构成，预测只因样本构成不同而产生的噪声估计之比。
source("/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/Question 2/fitlib.R")
suppressMessages(library(parallel))
nc <- as.integer(commandArgs(trailingOnly = TRUE)[1])
cts <- c("B cells", "Myeloid cells", "NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells")
one <- function(ct) {
  dat <- load_ct(ct); S <- dat$S; M <- dat$M
  sm <- relevel(factor(tag(S$sm_name)), ref = tag(DMSO))
  design <- model.matrix(~ 0 + sm + factor(S$donor_id) + factor(S$plate_name) + factor(S$row)); design <- design[, colSums(abs(design)) > 0, drop = FALSE]
  q <- qr(design); design <- design[, sort(q$pivot[seq_len(q$rank)]), drop = FALSE]
  v <- voom(calcNormFactors(DGEList(M)), design); fit <- lmFit(v, design)
  res2 <- (v$E - fit$coefficients %*% t(design))^2 * v$weights
  grp <- ifelse(S$sm_name == DMSO, "DMSO", ifelse(S$sm_name %in% PC, "PC", "ORD"))
  set.seed(1); gs <- sort(sample(which(rowSums(M) >= 50), 3000))             # 排除几乎无计数的基因
  D <- do.call(rbind, lapply(gs, function(g) { hw <- rowSums(qr.Q(qr(design * sqrt(v$weights[g, ])))^2)
    c(sapply(c("DMSO", "PC", "ORD"), function(k) sum(res2[g, grp == k]) / sum(1 - hw[grp == k])), sapply(c("DMSO", "PC", "ORD"), function(k) sum(1 - hw[grp == k]))) }))
  colnames(D) <- c("v_DMSO", "v_PC", "v_ORD", "df_DMSO", "df_PC", "df_ORD")
  data.frame(ct = ct, gene = rownames(M)[gs], logcount = fit$Amean[gs], D, n_DMSO = sum(grp == "DMSO"), n_PC = sum(grp == "PC"), n_ORD = sum(grp == "ORD"))
}
R <- do.call(rbind, mclapply(cts, one, mc.cores = min(nc, 6)))
write.csv(R, file.path(Q2, "out", "group_variance.csv"), row.names = FALSE); cat("DONE\n")
