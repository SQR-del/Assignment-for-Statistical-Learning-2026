"""Adapter: run the teammate branch's first-round models (those it submitted to Kaggle, branch REPORT.md section 12)
inside this repo's validation harness.

Models (configs of REPORT.md sections 5 and 12, i.e. before the section-13 re-tuning):
  pca_ridge    target-space PCA (128) + ridge alpha=10                          models_emb.m_pca_ridge
  mlp_sklearn  sklearn MLP (256,128), alpha=1e-3, early stopping                models_emb.m_mlp_sklearn
  mlp          torch full-output MLP (512,256), dropout 0.1, 150 epochs, wd 1e-4  nn_emb.train_full_mlp
  mlp_wide     same, (1024,512)
  ensemble_r1  equal mean of pca_ridge + ridge(10) + pls(20) + krr_rbf_sig(1) + knn(5)   (section-5 table)
  blend_r1     equal mean of mlp_wide, mlp, ensemble_r1, mlp_sklearn                     (section-12 submission.csv)

All of them take the feature dict ("view") of the branch's common.build_views (code/common_emb.py). The view is built
here from a frame of the table whose hidden rows are NaN; build_views only reads rows of fold.train_idx for its
signatures, so a leak would surface as NaN and fail. Choices made when porting (see methods/classic_lab/README.md):
  * training rows = every visible row of the table (also the 12 positive-control rows, which the branch dropped,
    and the other sparse cell type's anchors), as for 16-19;
  * anchors = the 15 non-control compounds observed in B and Myeloid cells, as in the branch;
  * every prediction is clipped to the per-gene range of the training labels, as the branch did;
  * torch models use the branch's single seed 20251001.
"""
import os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
import common_emb as CE
import models_emb as ME
sys.modules.setdefault("models", ME)        # nn_emb.train_full_mlp does `from models import _std_targets`
import nn_emb
from common import PROJ

SEED = 20251001
TORCH_MLP = dict(mlp=dict(hidden=(512, 256), epochs=150, wd=1e-4),
                 mlp_wide=dict(hidden=(1024, 512), epochs=150, wd=1e-4))
BLEND = ("mlp_wide", "mlp", "ensemble_r1", "mlp_sklearn")
ALL = ("pca_ridge", "mlp_sklearn", "mlp", "mlp_wide", "ensemble_r1", "blend_r1")
_C = {}


def frame(tab):
    """de_train-like frame (cell_type, drug, SMILES, genes) of the table rows; labels are tab.Y (hidden rows NaN)."""
    if "smi" not in _C:
        d = pd.read_parquet(os.path.join(PROJ, "data", "de_train.parquet"), columns=["sm_name", "SMILES"])
        d = d.astype(str).drop_duplicates("sm_name"); _C["smi"] = dict(zip(d.sm_name, d.SMILES))
    meta = pd.DataFrame(dict(cell_type=[tab.cells[c] for c in tab.c_row], drug=[tab.comps[d] for d in tab.d_row]))
    meta["SMILES"] = meta.drug.map(_C["smi"])
    df = pd.concat([meta, pd.DataFrame(tab.Y.astype(np.float32), columns=tab.genes)], axis=1)
    if "fp" not in _C:                                   # label-free SMILES fingerprint PCA over the compounds
        _C["fp"] = CE.drug_fingerprint_pca(df, n_comp=128)
    return df


def ensemble_r1(view):
    Xtr, Xte = ME._scaled(view); y = view["y_train"]
    parts = [ME.m_pca_ridge(view, alpha=10.0),
             Ridge(alpha=10.0).fit(Xtr, y).predict(Xte),
             PLSRegression(n_components=20, scale=False).fit(Xtr, y).predict(Xte),
             ME.m_krr_rbf_sig(view, alpha=1.0),
             KNeighborsRegressor(n_neighbors=5, weights="distance").fit(Xtr, y).predict(Xte)]
    return np.mean(parts, 0)


def predict_models(tab, vis, qry, which=ALL):
    """Train the requested models on rows `vis`, predict rows `qry` (one target cell type). {name: [len(qry), G]}."""
    vis = np.asarray(vis); qry = np.asarray(qry)
    t = np.unique(tab.c_row[qry]); assert len(t) == 1, "one target cell type per call"
    fold = CE.Fold(name="fold", protocol="harness", target_cell=tab.cells[t[0]],
                   anchors=[tab.comps[a] for a in tab.ANC], ref_cells=[], train_idx=vis, test_idx=qry, test_drugs=[])
    view = CE.build_views(frame(tab), tab.genes, fold, _C["fp"])
    ytr = view["y_train"]; assert np.isfinite(ytr).all(), "hidden rows among the training rows"
    lo, hi = ytr.min(0), ytr.max(0); out = {}
    need = set(which) | (set(BLEND) if "blend_r1" in which else set())
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        for m in [m for m in ALL if m in need and m != "blend_r1"]:
            if m == "pca_ridge":
                p = ME.m_pca_ridge(view, alpha=10.0)
            elif m == "mlp_sklearn":
                p = ME.m_mlp_sklearn(view, hidden=(256, 128), alpha=1e-3)
            elif m == "ensemble_r1":
                p = ensemble_r1(view)
            else:
                Xtr, Xte = ME._scaled(view)
                p = nn_emb.train_full_mlp(Xtr, Xte, ytr, dsig_tr=view["Dsig_train"], dsig_te=view["Dsig_test"],
                                          seed=SEED, **TORCH_MLP[m])
            out[m] = np.clip(np.asarray(p, np.float32), lo, hi)
    finally:
        torch.set_default_device("cpu")
    if "blend_r1" in need:
        out["blend_r1"] = np.clip(np.mean([out[k] for k in BLEND], 0), lo, hi)
    return {m: out[m] for m in which}


def harness_predict(name):
    """predict(tab, hidden, qry, seeds) for methods/harness.py (seeds is ignored: the branch used one seed)."""
    def predict(tab, hidden, qry, seeds):
        hs = set(np.asarray(hidden).tolist()); vis = [i for i in range(tab.N) if i not in hs]
        qry = np.asarray(qry); out = np.zeros((len(qry), tab.G))
        for t in np.unique(tab.c_row[qry]):              # export: B and Myeloid queries in one call
            pos = np.flatnonzero(tab.c_row[qry] == t)
            out[pos] = predict_models(tab, vis, qry[pos], (name,))[name]
        return out
    return predict
