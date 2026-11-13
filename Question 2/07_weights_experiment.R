# 07_weights_experiment.R  查清小模型与大模型之间、自由度以外的差别来自哪里
# 用法: Rscript 07_weights_experiment.R "<细胞类型>" <重复次数> <核数>      （重复用 01_sim.R 抽好的药：sim/<细胞>/draws.csv）
# A. 干预实验。对每次重复的小模型样本，做三种拟合：
#      std      小模型自己的 voom 权重和归一化（= 01_sim.R 的 small）
#      fullw    表达量用小模型的归一化，权重换成大模型 voom 给同一批样本的权重
#      fullwE   表达量和权重都取自大模型的 voom
#    若换权重后与大模型的差别只剩自由度，则权重就是自由度以外的全部原因。
# B. 两条 voom 曲线（表达量 -> 预期噪声）：大模型一条，每次重复的小模型一条。
# C. 大模型第一遍（不加权）拟合的残差，按样本组（DMSO / 阳性对照 / 普通药）分别算每个基因的残差标准差（按杠杆值校正）。
# D. 大模型最终加权拟合：各组"实际残差方差 / voom 预测方差"（随机 2000 个基因，按基因各自的杠杆值校正）。
source("/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/Question 2/fitlib.R")
suppressMessages(library(parallel))
a <- commandArgs(trailingOnly = TRUE); ct <- a[1]; R <- as.integer(a[2]); nc <- as.integer(a[3])
dat <- load_ct(ct); S <- dat$S; M <- dat$M
out <- file.path(Q2, "weights_exp", tag(ct)); dir.create(out, showWarnings = FALSE, recursive = TRUE)
XG <- seq(-2, 16, by = 0.25)                                        # 曲线取值的固定网格（log2 计数）

mk <- function(S) {
  sm <- relevel(factor(tag(S$sm_name)), ref = tag(DMSO))
  design <- model.matrix(~ 0 + sm + factor(S$donor_id) + factor(S$plate_name) + factor(S$row))
  colnames(design) <- make.names(colnames(design)); design <- design[, colSums(abs(design)) > 0, drop = FALSE]
  q <- qr(design); list(design = design[, sort(q$pivot[seq_len(q$rank)]), drop = FALSE], sm = sm)
}
run <- function(S, E, W) {                                          # 给定表达量和权重的拟合；输出结构同 fit_one
  d <- mk(S); design <- d$design; sm <- d$sm
  fit <- lmFit(E, design, weights = W)
  lv <- setdiff(levels(sm), tag(DMSO)); lv <- lv[paste0("sm", lv) %in% colnames(design)]
  cm <- makeContrasts(contrasts = paste0("sm", lv, " - sm", tag(DMSO)), levels = colnames(design))
  f2 <- eBayes(contrasts.fit(fit, cm))
  lp <- pt(abs(f2$t), f2$df.total, lower.tail = FALSE, log.p = TRUE) + log(2)
  nm <- setNames(unique(S$sm_name), tag(unique(S$sm_name)))
  list(slp = sign(f2$coefficients) * (-lp / log(10)), t = f2$t, coef = f2$coefficients, su = f2$stdev.unscaled, s2post = f2$s2.post, sigma = f2$sigma,
       compounds = unname(nm[lv]), wells = as.integer(table(S$sm_name)[unname(nm[lv])]),
       scal = data.frame(n_samples = nrow(S), rank = ncol(design), df_resid = median(f2$df.residual), df_prior = median(f2$df.prior), df_total = median(f2$df.total),
                         s2_prior = median(f2$s2.prior), n_dmso = sum(S$sm_name == DMSO), n_pc = sum(S$sm_name %in% PC)))
}
line_at <- function(v) approx(v$voom.line$x, v$voom.line$y, xout = XG, rule = 2)$y ^ 2      # voom 曲线给出 sqrt(sd)，平方得预期标准差

## ---------- 大模型 ----------
d0 <- calcNormFactors(DGEList(M)); des <- mk(S); design <- des$design
vf <- voom(d0, design, save.plot = TRUE)
grp <- ifelse(S$sm_name == DMSO, "DMSO", ifelse(S$sm_name %in% PC, "PC", "ORD"))
write.csv(data.frame(sample = S$sample, sm_name = S$sm_name, group = grp, n_cells = S$n_cells, lib_size = d0$samples$lib.size, norm_factor = d0$samples$norm.factors,
                     mean_weight = colMeans(vf$weights)), file.path(out, "samples_full.csv"), row.names = FALSE)
# C. 第一遍不加权拟合的分组残差
fit0 <- lmFit(vf$E, design); res <- vf$E - fit0$coefficients %*% t(design)
h <- rowSums(qr.Q(qr(design))^2)
sx <- fit0$Amean + mean(log2(d0$samples$lib.size * d0$samples$norm.factors + 1)) - log2(1e6)
gsd <- function(g) sqrt(rowSums(res[, grp == g, drop = FALSE]^2) / sum(1 - h[grp == g]))
G1 <- data.frame(sx = sx, sd_all = fit0$sigma, sd_DMSO = gsd("DMSO"), sd_PC = gsd("PC"), sd_ORD = gsd("ORD"))[ORD, ]
write.csv(G1, file.path(out, "firstpass_group_sd.csv"), row.names = FALSE)
cat("leverage sum by group:", tapply(h, grp, sum), "| n by group:", table(grp), "\n")
# D. 加权拟合：实际 / 预测 方差
ff <- lmFit(vf, design); resw <- (vf$E - ff$coefficients %*% t(design)) * sqrt(vf$weights)
set.seed(1); gs <- sort(sample(nrow(M), 2000))
D <- do.call(rbind, mclapply(gs, function(g) { hw <- rowSums(qr.Q(qr(design * sqrt(vf$weights[g, ])))^2)
  sapply(c("DMSO", "PC", "ORD"), function(k) sum(resw[g, grp == k]^2) / sum(1 - hw[grp == k])) }, mc.cores = nc))
write.csv(data.frame(gene = rownames(M)[gs], sx = sx[gs], D), file.path(out, "weighted_group_ratio.csv"), row.names = FALSE)
lines_out <- data.frame(x = XG, full = line_at(vf))

## ---------- 小模型（各次重复） ----------
draws <- read.csv(file.path(Q2, "sim", tag(ct), "draws.csv"), stringsAsFactors = FALSE)
res_small <- mclapply(0:R, function(r) {
  A <- draws$sm_name[draws$rep == r & draws$in_small]; idx <- S$sm_name %in% c(A, PC, DMSO)
  Ss <- S[idx, ]; d0s <- calcNormFactors(DGEList(M[, idx])); vs <- voom(d0s, mk(Ss)$design, save.plot = TRUE)
  save_fit(run(Ss, vs$E, vs$weights), file.path(out, sprintf("r%02d_std", r)), only = A)
  save_fit(run(Ss, vs$E, vf$weights[, idx]), file.path(out, sprintf("r%02d_fullw", r)), only = A)
  save_fit(run(Ss, vf$E[, idx], vf$weights[, idx]), file.path(out, sprintf("r%02d_fullwE", r)), only = A)
  # 同一批样本在两次 voom 中的归一化因子与权重之比
  list(line = line_at(vs), nf_ratio = median((d0s$samples$lib.size * d0s$samples$norm.factors) / (d0$samples$lib.size * d0$samples$norm.factors)[idx]))
}, mc.cores = nc, mc.preschedule = FALSE)
for (r in 0:R) lines_out[[sprintf("small_r%02d", r)]] <- res_small[[r + 1]]$line
write.csv(lines_out, file.path(out, "voom_lines.csv"), row.names = FALSE)
cat(ct, "effective library size small/full (median over reps):", median(sapply(res_small, function(z) z$nf_ratio)), "\nDONE\n")
