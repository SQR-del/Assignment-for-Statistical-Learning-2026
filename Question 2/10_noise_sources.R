# 10_noise_sources.R  为什么对照孔与普通药物孔的噪声水平不同：逐样本的残差统计 + 两个扩展模型
# 用法: Rscript 10_noise_sources.R <核数>      输出: out/noise_samples.csv
# 对 6 种细胞类型的训练样本，固定官方 voom 的表达量和权重，分别用三种均值模型拟合：
#   base     官方设计：药物 + 供体 + 板 + 行
#   col      在 base 上加"列"（孔在板上的列位置）           -> 检验假设：未建模的列效应
#   pcdonor  在 base 上加"阳性对照 x 供体"交互项           -> 检验假设：阳性对照在不同供体身上反应不同
# 对随机 3000 个基因，计算每个样本的 q_i = 对基因取平均的 w_gi * r_gi^2 / (1 - h_gi)
#   （r 为残差，w 为 voom 权重，h 为该基因加权设计的杠杆值；模型正确且各样本同样吵时 q 应都接近同一个值）
# 另按基因表达量高低各一半分别给出 q_low、q_high；qmed 为对基因取中位数的稳健版本（分析以它为准）。
source("/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/Question 2/fitlib.R")
suppressMessages(library(parallel))
nc <- as.integer(commandArgs(trailingOnly = TRUE)[1])
cts <- c("B cells", "Myeloid cells", "NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells")
reduce <- function(X) { X <- X[, colSums(abs(X)) > 0, drop = FALSE]; q <- qr(X); X[, sort(q$pivot[seq_len(q$rank)]), drop = FALSE] }
one <- function(ct) {
  dat <- load_ct(ct); S <- dat$S; M <- dat$M
  S$col <- as.integer(gsub("[^0-9]", "", S$well))
  sm <- relevel(factor(tag(S$sm_name)), ref = tag(DMSO))
  X0 <- model.matrix(~ 0 + sm + factor(S$donor_id) + factor(S$plate_name) + factor(S$row))
  Xcol <- cbind(X0, model.matrix(~ factor(S$col))[, -1, drop = FALSE])
  dn <- sort(unique(S$donor_id)); inter <- do.call(cbind, lapply(PC, function(p) sapply(dn[-1], function(d) as.numeric(S$sm_name == p & S$donor_id == d))))
  designs <- list(base = reduce(X0), col = reduce(Xcol), pcdonor = reduce(cbind(X0, inter)))
  v <- voom(calcNormFactors(DGEList(M)), designs$base)
  set.seed(1); gs <- sort(sample(which(rowSums(M) >= 50), min(3000, sum(rowSums(M) >= 50))))
  E <- v$E[gs, ]; W <- v$weights[gs, ]; lo <- rowMeans(E) <= median(rowMeans(E))
  out <- data.frame(ct = ct, sample = S$sample, sm_name = S$sm_name, group = ifelse(S$sm_name == DMSO, "DMSO", ifelse(S$sm_name %in% PC, "PC", "ORD")),
                    donor = S$donor_id, plate = S$plate_name, row = S$row, col = S$col, n_cells = S$n_cells, lib_size = S$lib_size)
  for (m in names(designs)) {
    X <- designs[[m]]
    Q <- sapply(seq_along(gs), function(k) { w <- W[k, ]; Xw <- X * sqrt(w); qrx <- qr(Xw); r <- qr.resid(qrx, E[k, ] * sqrt(w)); h <- rowSums(qr.Q(qrx)^2); r^2 / pmax(1 - h, 1e-8) })
    out[[paste0("q_", m)]] <- rowMeans(Q); out[[paste0("qlow_", m)]] <- rowMeans(Q[, lo]); out[[paste0("qhigh_", m)]] <- rowMeans(Q[, !lo])
    out[[paste0("qmed_", m)]] <- apply(Q, 1, median)          # 对基因取中位数：不受少数极端基因影响
    out[[paste0("rank_", m)]] <- ncol(X)
  }
  out
}
R <- do.call(rbind, mclapply(cts, one, mc.cores = min(nc, 6)))
write.csv(R, file.path(Q2, "out", "noise_samples.csv"), row.names = FALSE); cat("DONE\n")
