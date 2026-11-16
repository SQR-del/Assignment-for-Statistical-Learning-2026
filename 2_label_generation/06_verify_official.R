# 06_verify_official.R  用主办方的设置在全部训练样本上拟合 6 种细胞类型，供 06_verify_official.py 与官方标签比对
# 用法: Rscript 06_verify_official.R
source("/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/2_label_generation/fitlib.R")
suppressMessages(library(parallel))
cts <- c("B cells", "Myeloid cells", "NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells")
res <- mclapply(cts, function(ct) { f <- fit_one(load_ct(ct)); save_fit(f, file.path(Q2, "official_fit", tag(ct)))
  sprintf("%-20s n=%d rank=%d df_resid=%.0f df_prior=%.2f", ct, f$scal$n_samples, f$scal$rank, f$scal$df_resid, f$scal$df_prior) }, mc.cores = 6)
cat(paste(unlist(res), collapse = "\n"), "\nDONE\n")
