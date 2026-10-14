.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({library(limma); library(edgeR); library(Matrix); library(parallel)})
NCORES <- as.integer(Sys.getenv("DE_CORES", "12"))
args <- commandArgs(trailingOnly = TRUE); variant <- args[1]; out <- args[2]
only <- if (length(args) > 2 && nchar(args[3]) > 0) strsplit(args[3], "|", fixed = TRUE)[[1]] else NULL   # restrict compounds (simulation)
only_ct <- if (length(args) > 3) strsplit(args[4], "|", fixed = TRUE)[[1]] else NULL                    # ...only within these cell types
D <- "/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/results/pseudobulk_train"
M <- as(readMM(file.path(D, "counts.mtx")), "CsparseMatrix"); genes <- read.csv(file.path(D, "genes.csv"))$gene
S <- read.csv(file.path(D, "samples.csv"), stringsAsFactors = FALSE); de_genes <- read.csv(file.path(D, "de_genes.csv"))$gene
rownames(M) <- genes; colnames(M) <- S$sample
if (grepl("g18", variant)) M <- M[de_genes, ]
keep <- colSums(M) > 0
if (!is.null(only)) keep <- keep & (!(S$cell_type %in% only_ct) | S$sm_name %in% c(only, "Dimethyl Sulfoxide"))
S <- S[keep, ]; M <- M[, keep]
tr <- function(x) gsub("[^[:alnum:]]", "_", x)
S$smct <- tr(paste0(S$sm_name, "_", S$cell_type))
d0 <- calcNormFactors(DGEList(as.matrix(M)))
smct <- factor(S$smct)
cov <- if (grepl("lib", variant)) factor(tr(S$library_id)) else factor(tr(S$plate_name))
design <- model.matrix(~ 0 + smct + cov)
q <- qr(design); design <- design[, q$pivot[seq_len(q$rank)], drop = FALSE]
cat("design", dim(design), "\n")
v <- voom(d0, design); cat("voom done\n")
pairs <- unique(S[S$sm_name != "Dimethyl Sulfoxide", c("cell_type", "sm_name")])
lab <- paste0("smct", tr(paste0(pairs$sm_name, "_", pairs$cell_type)))
ctl <- paste0("smct", tr(paste0("Dimethyl Sulfoxide_", pairs$cell_type)))
cm <- makeContrasts(contrasts = paste0(lab, " - ", ctl), levels = colnames(design))
chunks <- split(seq_len(nrow(v)), cut(seq_len(nrow(v)), NCORES, labels = FALSE))
parts <- mclapply(chunks, function(ix) contrasts.fit(lmFit(v[ix, ], design), cm), mc.cores = NCORES)
fit <- parts[[1]]
for (nm in c("coefficients", "stdev.unscaled")) fit[[nm]] <- do.call(rbind, lapply(parts, function(p) p[[nm]]))
for (nm in c("sigma", "df.residual", "Amean")) fit[[nm]] <- unlist(lapply(parts, function(p) p[[nm]]))
fit$genes <- NULL; cat("lmFit done (parallel)\n")
f2 <- eBayes(fit, robust = grepl("robust", variant))
slp <- (sign(f2$coefficients) * -log10(f2$p.value))[de_genes, , drop = FALSE]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
idx <- data.frame(cell_type = pairs$cell_type, sm_name = pairs$sm_name, file = "all.bin", col = seq_len(nrow(pairs)))
writeBin(as.vector(slp), file.path(out, "all.bin")); writeBin(as.vector((f2$t)[de_genes, ]), file.path(out, "all.t.bin"))
write.csv(idx, file.path(out, "index.csv"), row.names = FALSE)
cat("df.prior", f2$df.prior[1], "df.resid", median(f2$df.residual), "\nDONE\n")
