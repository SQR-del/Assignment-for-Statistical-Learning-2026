"""Adapter: run the teammate branch's networks (code/nn_emb.py) inside this repo's validation harness.

The branch trainers take integer (cell, drug) ids, the training labels and the test ids. Here those are built from a
common.Table whose hidden rows are NaN: only visible rows are passed on, so a leak would surface as NaN and fail.
Choices made when porting (see methods/emb_lab/README.md):
  * the 12 positive-control rows are kept as training rows (the branch dropped them);
  * predictions are clipped to the per-gene range of the training labels, as the branch harness did;
  * seeds: base 20251001 + r as in the branch, number of seeds = the `seeds` argument of the harness;
  * hyper-parameters are the branch defaults after its tuning (emb 128, hidden 256-512-1024, K=25, 400 epochs,
    lr 3e-3, dropout 0.2).
"""
import os
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
import nn_emb
from common import PROJ

BASE_SEED = 20251001
FULL_CELLS = ("T cells CD4+", "T regulatory cells", "T cells CD8+", "NK cells")
BLEND_B = dict(embed=0.50, smiles=0.30, hybrid=0.20)          # the final blend of the branch (its methods/13_final_blend)
_SM = {}


def smiles_codes(tab):
    """Per-compound SMILES character codes, built with build_smiles from the branch."""
    if not _SM:
        df = pd.read_parquet(os.path.join(PROJ, "data", "de_train.parquet"), columns=["sm_name", "SMILES"])
        df = df.astype(str).drop_duplicates("sm_name").rename(columns={"sm_name": "drug"})      # columns are categorical on disk
        d2s, vocab = nn_emb.build_smiles(df)
        _SM.update(d2s=d2s, vocab=vocab)
    return np.stack([_SM["d2s"][d] for d in tab.comps]), _SM["vocab"]


def drug_signature(tab, rows, visible):
    """Branch feature `dsig`: mean response of the compound of each row in the OTHER T/NK cell types, visible rows only."""
    full = [tab.CI[c] for c in FULL_CELLS]; out = np.zeros((len(rows), tab.G), np.float32)
    for k, i in enumerate(rows):
        c, d = tab.c_row[i], tab.d_row[i]
        src = [tab.row_of[(o, d)] for o in full if o != c and (o, d) in tab.row_of and visible[tab.row_of[(o, d)]]]
        if src: out[k] = tab.Y[src].mean(0)
    return out


def predict_models(tab, vis, qry, seeds, which=("embed", "hybrid", "smiles")):
    """Train the requested branch models on rows `vis`, predict rows `qry`. Returns {name: [len(qry), G]}."""
    vis = np.asarray(vis); qry = np.asarray(qry)
    x = np.stack([tab.c_row, tab.d_row], 1).astype(np.int64)
    ytr = tab.Y[vis].astype(np.float32); assert np.isfinite(ytr).all(), "hidden rows among the training rows"
    nc, nd = len(tab.cells), len(tab.comps); lo, hi = ytr.min(0), ytr.max(0); out = {}
    med = lambda f: np.clip(np.median(np.stack([f(BASE_SEED + r) for r in range(seeds)]), 0), lo, hi)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if "embed" in which:
            out["embed"] = med(lambda s: nn_emb.train_embed(x[vis], ytr, x[qry], nc, nd, seed=s))
        if "hybrid" in which:
            visible = np.zeros(tab.N, bool); visible[vis] = True
            Dtr, Dte = drug_signature(tab, vis, visible), drug_signature(tab, qry, visible)
            p = PCA(n_components=64, random_state=BASE_SEED).fit(Dtr)
            Xtr, Xte = p.transform(Dtr).astype("float32"), p.transform(Dte).astype("float32")
            out["hybrid"] = med(lambda s: nn_emb.train_hybrid(x[vis], Xtr, ytr, x[qry], Xte, nc, nd, seed=s))
        if "smiles" in which:
            sym, vocab = smiles_codes(tab); dv, dq = tab.d_row[vis], tab.d_row[qry]
            out["smiles"] = med(lambda s: nn_emb.train_smiles(x[vis], sym[dv], ytr, x[qry], sym[dq], vocab, nc, nd, seed=s))
    finally:
        torch.set_default_device("cpu")
    if all(k in out for k in BLEND_B):
        out["blend_b"] = sum(w * out[k] for k, w in BLEND_B.items())
    return out


def harness_predict(name):
    """predict(tab, hidden, qry, seeds) for methods/harness.py; name in embed | hybrid | smiles | blend_b."""
    def predict(tab, hidden, qry, seeds):
        hs = set(np.asarray(hidden).tolist()); vis = [i for i in range(tab.N) if i not in hs]
        which = tuple(BLEND_B) if name == "blend_b" else (name,)
        return predict_models(tab, vis, qry, seeds, which)[name].astype(np.float64)
    return predict
