.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({library(limma); library(edgeR); library(Matrix)})
args <- commandArgs(trailingOnly = TRUE); variant <- args[1]; out <- args[2]
only <- if (length(args) > 2) strsplit(args[3], ",")[[1]] else NULL          # optional: restrict compounds (simulation)
drop_ct <- if (length(args) > 3) args[4] else ""                              # cell type the restriction applies to ("" = all)
D <- "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/results/pseudobulk_train"
M <- as(readMM(file.path(D, "counts.mtx")), "CsparseMatrix"); genes <- read.csv(file.path(D, "genes.csv"))$gene
S <- read.csv(file.path(D, "samples.csv"), stringsAsFactors = FALSE); de_genes <- read.csv(file.path(D, "de_genes.csv"))$gene
rownames(M) <- genes; colnames(M) <- S$sample
if (grepl("g18", variant)) M <- M[de_genes, ]
tr <- function(x) gsub("[^[:alnum:]]", "_", x)
dir.create(out, showWarnings = FALSE, recursive = TRUE); index <- data.frame()
for (ct in sort(unique(S$cell_type))) {
  idx <- S$cell_type == ct & colSums(M) > 0
  if (!is.null(only) && (drop_ct == "" || drop_ct == ct)) idx <- idx & S$sm_name %in% c(only, "Dimethyl Sulfoxide")
  Sc <- S[idx, ]; Mc <- as.matrix(M[, idx])
  d0 <- calcNormFactors(DGEList(Mc))
  sm <- factor(tr(Sc$sm_name)); sm <- relevel(sm, ref = tr("Dimethyl Sulfoxide"))
  cov <- if (grepl("lib", variant)) factor(tr(Sc$library_id)) else factor(tr(Sc$plate_name))
  design <- model.matrix(~ 0 + sm + cov)
  design <- design[, colSums(abs(design)) > 0, drop = FALSE]
  q <- qr(design); design <- design[, q$pivot[seq_len(q$rank)], drop = FALSE]    # drop aliased covariate columns
  v <- voom(d0, design); fit <- lmFit(v, design)
  lv <- setdiff(levels(sm), tr("Dimethyl Sulfoxide")); names_map <- setNames(unique(Sc$sm_name), tr(unique(Sc$sm_name)))
  cm <- makeContrasts(contrasts = paste0("sm", lv, " - sm", tr("Dimethyl Sulfoxide")), levels = colnames(design))
  f2 <- eBayes(contrasts.fit(fit, cm), robust = grepl("robust", variant))
  slp <- sign(f2$coefficients) * -log10(f2$p.value)
  slp <- slp[de_genes, , drop = FALSE]; tt <- (f2$t)[de_genes, , drop = FALSE]
  fn <- file.path(out, paste0(tr(ct), ".bin")); writeBin(as.vector(slp), fn)          # column-major genes x compounds
  writeBin(as.vector(tt), file.path(out, paste0(tr(ct), ".t.bin")))
  index <- rbind(index, data.frame(cell_type = ct, sm_name = unname(names_map[lv]), file = basename(fn), col = seq_along(lv),
                                   df_prior = f2$df.prior[1], df_resid = median(f2$df.residual), n_samples = sum(idx)))
  cat(ct, "done: samples", sum(idx), "compounds", length(lv), "df.resid", median(f2$df.residual), "df.prior", round(f2$df.prior[1], 2), "\n")
}
write.csv(index, file.path(out, "index.csv"), row.names = FALSE); cat("DONE\n")
