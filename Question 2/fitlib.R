# fitlib.R  limma 拟合的公共函数。每次拟合针对一种细胞类型。
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({library(limma); library(edgeR)})
P <- "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction"
Q2 <- file.path(P, "Question 2")
DMSO <- "Dimethyl Sulfoxide"; PC <- c("Belinostat", "Dabrafenib")
tag <- function(x) gsub("[^[:alnum:]]", "_", x)
# 数据来自 00_prepare_official.py（按主办方步骤重建的伪 bulk）
GEN_OFF <- readLines(file.path(Q2, "cache_official", "genes.txt"))
GEN_DE <- readLines(file.path(Q2, "cache_official", "genes_de_train_order.txt"))          # de_train.parquet 的基因列顺序
ORD <- match(GEN_DE, GEN_OFF)                                      # 存盘时统一换成 de_train 的顺序
load_ct <- function(ct) {
  S <- read.csv(file.path(Q2, "cache_official", paste0(tag(ct), "_samples.csv")), stringsAsFactors = FALSE)
  M <- matrix(readBin(file.path(Q2, "cache_official", paste0(tag(ct), ".i32")), "integer", n = nrow(S) * length(GEN_OFF), size = 4), nrow = length(GEN_OFF))
  rownames(M) <- GEN_OFF; colnames(M) <- S$sample
  list(M = M, S = S)
}

# keep_sm: 保留哪些药（NULL = 全部）；DMSO 总是保留
# 设置与主办方一致：TMM 归一化 -> voom -> lmFit -> 对 DMSO 的对比 -> eBayes（非稳健）
fit_one <- function(dat, keep_sm = NULL) {
  S <- dat$S; M <- dat$M
  idx <- rep(TRUE, nrow(S)); if (!is.null(keep_sm)) idx <- S$sm_name %in% c(keep_sm, DMSO)
  S <- S[idx, ]; M <- M[, idx]
  d0 <- calcNormFactors(DGEList(M))
  sm <- relevel(factor(tag(S$sm_name)), ref = tag(DMSO))
  # 主办方 compute_de.ipynb 的设计：~0 + 药物 + donor_id + plate_name + row（供体被板包含，共线列在下面去掉）
  design <- model.matrix(~ 0 + sm + factor(S$donor_id) + factor(S$plate_name) + factor(S$row))
  colnames(design) <- make.names(colnames(design))
  design <- design[, colSums(abs(design)) > 0, drop = FALSE]
  q <- qr(design); design <- design[, sort(q$pivot[seq_len(q$rank)]), drop = FALSE]   # 去掉共线的协变量列
  v <- voom(d0, design); fit <- lmFit(v, design)
  lv <- setdiff(levels(sm), tag(DMSO)); lv <- lv[paste0("sm", lv) %in% colnames(design)]
  cm <- makeContrasts(contrasts = paste0("sm", lv, " - sm", tag(DMSO)), levels = colnames(design))
  f2 <- eBayes(contrasts.fit(fit, cm))
  lp <- pt(abs(f2$t), f2$df.total, lower.tail = FALSE, log.p = TRUE) + log(2)          # 对数 p，避免下溢
  nm <- setNames(unique(S$sm_name), tag(unique(S$sm_name)))
  list(slp = sign(f2$coefficients) * (-lp / log(10)), t = f2$t, coef = f2$coefficients, su = f2$stdev.unscaled,
       s2post = f2$s2.post, sigma = f2$sigma, compounds = unname(nm[lv]),
       wells = as.integer(table(S$sm_name)[unname(nm[lv])]),
       scal = data.frame(n_samples = nrow(S), rank = ncol(design), df_resid = median(f2$df.residual),
                         df_prior = median(f2$df.prior), df_total = median(f2$df.total), s2_prior = median(f2$s2.prior),
                         n_dmso = sum(S$sm_name == DMSO), n_pc = sum(S$sm_name %in% PC)))
}
# 存盘：矩阵按 float32、列优先（基因 x 药）写出；python 端用 reshape(K, G) 读回
save_fit <- function(f, out, only = NULL, light = FALSE) {
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  k <- if (is.null(only)) seq_along(f$compounds) else which(f$compounds %in% only)
  w <- function(x, nm) writeBin(as.vector(if (is.matrix(x)) x[ORD, , drop = FALSE] else x[ORD]), file.path(out, nm), size = 4)
  w(f$slp[, k, drop = FALSE], "slp.f32"); w(f$t[, k, drop = FALSE], "t.f32")
  writeBin(as.vector(f$slp[ORD, k, drop = FALSE]), file.path(out, "slp.f64"), size = 8)                   # 双精度，供逐数值比对
  if (!light) { w(f$coef[, k, drop = FALSE], "coef.f32"); w(f$su[, k, drop = FALSE], "su.f32"); w(f$s2post, "s2post.f32"); w(f$sigma, "sigma.f32") }
  write.csv(data.frame(sm_name = f$compounds[k], wells = f$wells[k]), file.path(out, "compounds.csv"), row.names = FALSE)
  write.csv(f$scal, file.path(out, "scalars.csv"), row.names = FALSE)
}
