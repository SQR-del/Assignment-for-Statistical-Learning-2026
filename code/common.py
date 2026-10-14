"""Shared data / feature / metric utilities for the single-cell perturbation benchmark.

Targets are Limma signed -log10(p) statistics: y = -log10(p) * sign(LFC),
shape = (#cell_type x drug rows, 18211 genes).

Two evaluation protocols (see REPORT.md):
  P1  "competition-mimic": for a target cell type only the 15 anchor drugs are
      observed; predict the remaining 129 drugs (observed in the other cell types).
      Mirrors the real Kaggle task for B / Myeloid cells.
  P2  "leave-one-cell-type-out": target cell type has NO observed drug; train on
      the other 3 full cell types.  Pure drug-feature generalisation stress test.

Data location: the raw competition files (de_train.parquet, id_map.csv,
sample_submission.csv) are NOT shipped with the repo.  Put them in
    <project_root>/data/
or point the environment variable SCP_DATA_DIR at the folder that contains them.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("SCP_DATA_DIR", os.path.join(ROOT, "data"))
RESULTS_DIR = os.path.join(ROOT, "results")

FULL_CELLS = ["T cells CD4+", "T regulatory cells", "T cells CD8+", "NK cells"]
ALL_CELLS = FULL_CELLS + ["B cells", "Myeloid cells"]
ANCHOR_CELLS = ["B cells", "Myeloid cells"]
SEED = 20251001


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)


def results_path(*parts):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, *parts)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_de(drop_controls: bool = True):
    """Load de_train.parquet, rename sm_name->drug, drop the 12 control rows."""
    df = pd.read_parquet(data_path("de_train.parquet"))
    df = df.rename(columns={"sm_name": "drug"})
    if drop_controls:
        df = df[~df["control"]].reset_index(drop=True)
    genes = [c for c in df.columns if c not in
             ["cell_type", "drug", "sm_lincs_id", "SMILES", "control"]]
    return df, genes


def anchor_drugs(df):
    """The 15 drugs observed for B / Myeloid cells (used as the sparse anchor set)."""
    a = set(df[(df.cell_type == "B cells")].drug)
    b = set(df[(df.cell_type == "Myeloid cells")].drug)
    return sorted(a | b)


def cat_indices(df):
    """Integer codes for every cell type and drug (sorted), for embedding models."""
    cells = sorted(df.cell_type.unique())
    drugs = sorted(df.drug.unique())
    cid = {c: i for i, c in enumerate(cells)}
    did = {d: i for i, d in enumerate(drugs)}
    return cells, drugs, cid, did


# --------------------------------------------------------------------------- #
# chemical fingerprints (dependency-free hashed SMILES character n-grams)
# --------------------------------------------------------------------------- #
def smiles_fingerprints(smiles: pd.Series, n_bits: int = 2048):
    """Hashed character n-gram fingerprint of SMILES (1..3-grams).

    rdkit's Morgan fingerprint is unavailable in this environment (DLL load
    failure), so we use a deterministic, label-free string kernel instead.
    """
    vec = HashingVectorizer(analyzer="char", ngram_range=(1, 3),
                            n_features=n_bits, norm=None, alternate_sign=False)
    return vec.transform(smiles.fillna("").astype(str)).toarray().astype(np.float32)


def drug_fingerprint_pca(df: pd.DataFrame, n_comp: int = 128) -> pd.DataFrame:
    """Label-free PCA of SMILES fingerprints per unique drug (no leakage)."""
    uniq = df[["drug", "SMILES"]].drop_duplicates("drug").reset_index(drop=True)
    fp = smiles_fingerprints(uniq["SMILES"])
    k = min(n_comp, fp.shape[1], fp.shape[0] - 1)
    z = PCA(n_components=k, random_state=SEED).fit_transform(fp)
    cols = [f"fp_{i}" for i in range(k)]
    out = pd.DataFrame(z, columns=cols)
    out["drug"] = uniq["drug"].values
    return out


# --------------------------------------------------------------------------- #
# fold definitions
# --------------------------------------------------------------------------- #
@dataclass
class Fold:
    name: str
    protocol: str
    target_cell: str
    anchors: list
    ref_cells: list
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_drugs: list
    meta: dict = field(default_factory=dict)


def make_folds(df: pd.DataFrame, genes, anchors, protocol: str):
    """Return list of Fold objects.

    P1: target in FULL_CELLS; train = all rows of the other full cells + anchor
        rows of target; test = non-anchor rows of target.
    P2: target in FULL_CELLS; train = all rows of other full cells; test = all
        rows of target.  cell-type signature unavailable.
    """
    folds = []
    for c in FULL_CELLS:
        refs = [o for o in FULL_CELLS if o != c]
        if protocol == "P1":
            tr_mask = (df.cell_type.isin(refs)) | ((df.cell_type == c) & (df.drug.isin(anchors)))
            te_mask = (df.cell_type == c) & (~df.drug.isin(anchors))
        elif protocol == "P2":
            tr_mask = df.cell_type.isin(refs)
            te_mask = df.cell_type == c
        else:
            raise ValueError(protocol)
        tr = df.index[tr_mask].to_numpy()
        te = df.index[te_mask].to_numpy()
        folds.append(Fold(
            name=f"{protocol}_{c}", protocol=protocol, target_cell=c,
            anchors=list(anchors), ref_cells=refs,
            train_idx=tr, test_idx=te,
            test_drugs=sorted(df.loc[te, "drug"].unique()),
            meta={"n_train": len(tr), "n_test": len(te)},
        ))
    return folds


def inner_folds(df, genes, anchors, outer: Fold):
    """Nested validation folds: hold out one *reference* cell type as the inner
    target (with its 15 anchors), exactly mirroring the outer task.  The outer
    target's non-anchor drugs never enter the inner training set."""
    out = []
    refs = outer.ref_cells
    for c_in in refs:
        inner_refs = [o for o in refs if o != c_in]
        tr = df.index[df.cell_type.isin(inner_refs)
                      | ((df.cell_type == c_in) & (df.drug.isin(anchors)))
                      | ((df.cell_type == outer.target_cell) & (df.drug.isin(anchors)))]
        te = df.index[(df.cell_type == c_in) & (~df.drug.isin(anchors))]
        out.append(Fold(name=f"{outer.name}<-{c_in}", protocol="inner", target_cell=c_in,
                        anchors=list(anchors), ref_cells=inner_refs,
                        train_idx=tr.to_numpy(), test_idx=te.to_numpy(),
                        test_drugs=sorted(df.loc[te, "drug"].unique())))
    return out


# --------------------------------------------------------------------------- #
# leakage-safe signature features
# --------------------------------------------------------------------------- #
def _mean_vec(Y, idx):
    return Y[idx].mean(0) if len(idx) else np.zeros(Y.shape[1], dtype=np.float32)


def build_views(df: pd.DataFrame, genes, fold: Fold, fp_pca: pd.DataFrame,
                n_dsig: int = 64, n_csig: int = 64):
    """Build per-row features (leakage-safe: signatures use ONLY training rows).

    Returns dict with:
      Xc_train/Xc_test  : compact features (~256 dims) [fp | dsig | csig | scale]
      Dsig_train/test   : leave-own-cell-type-out drug signature
      Csig_train/test   : anchor-based cell-type signature
      y_train/y_test    : targets
      anchor_mask_train : rows of the target cell's anchor drugs in the train set
      rows_train/test   : (cell_type, drug) tuples
    """
    Y = df[genes].to_numpy(dtype=np.float32)
    anchors = set(fold.anchors)
    train_idx, test_idx = fold.train_idx, fold.test_idx
    # only rows visible in the training fold may contribute to signatures;
    # the held-out target cell's non-anchor drugs are NOT visible.
    visible = np.zeros(len(df), dtype=bool)
    visible[train_idx] = True

    def _lookup(cell, drug):
        hit = df.index[(df.cell_type == cell) & (df.drug == drug)]
        if len(hit) and visible[hit[0]]:
            return Y[hit[0]]
        return None

    def drug_sig_row(i):
        c = df.at[i, "cell_type"]
        d = df.at[i, "drug"]
        src = [o for o in FULL_CELLS if o != c]          # never the row's own cell type
        vecs = [v for v in (_lookup(o, d) for o in src) if v is not None]
        return np.mean(vecs, 0) if vecs else np.zeros(Y.shape[1], dtype=np.float32)

    def cell_sig(c, exclude_drug=None):
        """Mean over the 15 anchor drugs observed in cell type c (exclude a drug to avoid leakage)."""
        sub = df[(df.cell_type == c) & (df.drug.isin(anchors))]
        if exclude_drug is not None:
            sub = sub[sub.drug != exclude_drug]
        sub = sub[np.array([visible[i] for i in sub.index], dtype=bool)]
        if len(sub) == 0:
            return np.zeros(Y.shape[1], dtype=np.float32)
        return Y[sub.index].mean(0)

    def build(idx):
        ds, cs = [], []
        for i in idx:
            ds.append(drug_sig_row(i))
            cs.append(cell_sig(df.at[i, "cell_type"], exclude_drug=df.at[i, "drug"]))
        return np.asarray(ds, dtype=np.float32), np.asarray(cs, dtype=np.float32)

    Dtr, Ctr = build(train_idx)
    Dte, Cte = build(test_idx)

    # fold-local PCA on signatures, fit on training rows only
    kd = min(n_dsig, Dtr.shape[0] - 1, Dtr.shape[1])
    pd_ = PCA(n_components=kd, random_state=SEED).fit(Dtr)
    Dtr_p, Dte_p = pd_.transform(Dtr), pd_.transform(Dte)
    src_cells = sorted(set(df.loc[train_idx, "cell_type"]))
    cs_mat = np.stack([cell_sig(c) for c in src_cells]) if src_cells else np.zeros((1, Y.shape[1]))
    cs_all = np.vstack([cs_mat, cell_sig(fold.target_cell).reshape(1, -1)])
    kc = min(n_csig, cs_all.shape[0] - 1)
    pc_ = PCA(n_components=max(kc, 1), random_state=SEED).fit(cs_all) if cs_all.shape[0] > 1 else None
    if pc_ is not None:
        Ctr_p, Cte_p = pc_.transform(Ctr), pc_.transform(Cte)
    else:
        Ctr_p = np.zeros((len(train_idx), 1), np.float32); Cte_p = np.zeros((len(test_idx), 1), np.float32)

    fp_tr = fp_pca.set_index("drug").loc[df.loc[train_idx, "drug"].values].to_numpy(np.float32)
    fp_te = fp_pca.set_index("drug").loc[df.loc[test_idx, "drug"].values].to_numpy(np.float32)

    def compact(F, D, C):
        scale = np.stack([np.abs(D).mean(1), np.abs(C).mean(1)], 1)
        return np.concatenate([F, D, C, scale], 1).astype(np.float32)

    Xc_tr = compact(fp_tr, Dtr_p, Ctr_p)
    Xc_te = compact(fp_te, Dte_p, Cte_p)

    tc_train = df.loc[train_idx, "cell_type"].to_numpy()
    dr_train = df.loc[train_idx, "drug"].to_numpy()
    anchor_mask = (tc_train == fold.target_cell) & np.isin(dr_train, list(anchors))

    return {
        "Xc_train": Xc_tr, "Xc_test": Xc_te,
        "Dsig_train": Dtr, "Dsig_test": Dte,
        "Csig_train": Ctr, "Csig_test": Cte,
        "y_train": Y[train_idx], "y_test": Y[test_idx],
        "anchor_mask_train": anchor_mask,
        "anchor_mask_test": np.zeros(len(test_idx), bool),
        "rows_train": list(zip(df.loc[train_idx, "cell_type"], df.loc[train_idx, "drug"])),
        "rows_test": list(zip(df.loc[test_idx, "cell_type"], df.loc[test_idx, "drug"])),
        "genes": genes,
    }


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, genes, train_var=None, top_k: int = 1000):
    """Official metric is `mrrmse` (mean row-wise RMSE, lower is better)."""
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    resid = y_true - y_pred
    mse = float(np.mean(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(resid)))

    def _corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / denom) if denom > 0 else np.nan

    def _rank(x):
        order = np.argsort(x, axis=1)
        ranks = np.empty_like(order, dtype=np.float64)
        rows = np.arange(x.shape[0])[:, None]
        ranks[rows, order] = np.arange(x.shape[1])[None, :]
        return ranks

    pear = np.array([_corr(y_true[i], y_pred[i]) for i in range(len(y_true))])
    rt, rp = _rank(y_true), _rank(y_pred)
    spear = np.array([_corr(rt[i], rp[i]) for i in range(len(y_true))])
    # official competition metric: mean ROW-WISE RMSE (each sample weighted equally)
    mrrmse = float(np.mean(np.sqrt(np.mean(resid ** 2, axis=1))))
    m = {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae, "r2": r2,
         "mrrmse": mrrmse,
         "pearson_mean": float(np.nanmean(pear)),
         "pearson_median": float(np.nanmedian(pear)),
         "spearman_mean": float(np.nanmean(spear))}
    if train_var is not None:
        top = np.argsort(train_var)[::-1][:top_k]
        t = y_true[:, top]; p = y_pred[:, top]
        mse_t = float(np.mean((t - p) ** 2))
        ss = float(np.sum((t - t.mean()) ** 2))
        m[f"mse_top{top_k}"] = mse_t
        m[f"r2_top{top_k}"] = float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else np.nan
    return m, pear


def mrrmse(pred, truth):
    return float(np.mean(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2, axis=1))))


def train_gene_variance(y_train):
    return np.nanvar(y_train, axis=0)
