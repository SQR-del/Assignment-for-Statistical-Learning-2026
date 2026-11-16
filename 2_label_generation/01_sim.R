# 01_sim.R  随机抽药的小模型 / 大模型模拟
# 用法: Rscript 01_sim.R "<细胞类型>" <重复次数> <核数>
# 每种细胞类型：
#   full        全部样本（对应测试标签的拟合）
#   full_noPC   全部样本去掉两种阳性对照
#   每次重复 r：随机抽 15 种普通药 A_r，再补到 60 种 B_r（A_r 包含于 B_r）
#     small_r       DMSO + 阳性对照 + A_r   （对应 B/髓系训练标签的拟合）
#     small_noPC_r  DMSO + A_r              （样本少，但没有阳性对照）
#     mid_r         DMSO + 阳性对照 + B_r   （中等规模；只保存 A_r 的行）
#   r = 0 用真实的 15 种锚点药
source("/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/2_label_generation/fitlib.R")
suppressMessages(library(parallel))
a <- commandArgs(trailingOnly = TRUE); ct <- a[1]; R <- as.integer(a[2]); nc <- as.integer(a[3])
dat <- load_ct(ct); out <- file.path(Q2, "sim", tag(ct))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
ord <- setdiff(sort(unique(dat$S$sm_name)), c(DMSO, PC))
anch <- intersect(readLines(file.path(Q2, "cache_official", "anchors.txt")), ord)
set.seed(20261113 + sum(utf8ToInt(ct)))
draws <- lapply(0:R, function(r) { A <- if (r == 0) anch else sample(ord, 15)
  list(r = r, A = A, B = c(A, sample(setdiff(ord, A), 60 - length(A)))) })
write.csv(do.call(rbind, lapply(draws, function(d) data.frame(rep = d$r, sm_name = d$B, in_small = d$B %in% d$A))),
          file.path(out, "draws.csv"), row.names = FALSE)
tasks <- c(list(list(name = "full", keep = NULL, only = NULL), list(name = "full_noPC", keep = ord, only = NULL)),
  unlist(lapply(draws, function(d) list(
    list(name = sprintf("small_r%02d", d$r), keep = c(d$A, PC), only = NULL),
    list(name = sprintf("small_noPC_r%02d", d$r), keep = d$A, only = NULL),
    list(name = sprintf("mid_r%02d", d$r), keep = c(d$B, PC), only = d$A))), recursive = FALSE))
res <- mclapply(tasks, function(tk) { t0 <- Sys.time(); f <- fit_one(dat, keep_sm = tk$keep); save_fit(f, file.path(out, tk$name), only = tk$only)
  sprintf("%s n=%d df=%.1f %.0fs", tk$name, f$scal$n_samples, f$scal$df_total, as.numeric(difftime(Sys.time(), t0, units = "secs"))) },
  mc.cores = nc, mc.preschedule = FALSE)
cat(ct, "\n", paste(unlist(res), collapse = "\n"), "\nDONE\n")
