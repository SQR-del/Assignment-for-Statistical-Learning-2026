"""Leakage-safe feature construction for the real B / Myeloid test pairs.

This is a vectorised equivalent of ``common.build_views`` for a *test* fold made
of the Kaggle ``id_map.csv`` rows.  The original ``build_target_context`` in
``code/predict.py`` omitted the cell signature (``csig``) and the two scalar
scale features, so classic models received a different feature matrix than in
validation.  That is the main reason the old reproduction attempts did not
match the archived submissions.

The function below reproduces the validation feature matrix exactly:

    Xc = [fingerprint_pca(128) | dsig_pca(64) | csig_pca(k) | 2 scales]

where every PCA is fitted on the training rows (the target cell's non-anchor
drugs are never visible).  ``verify_against_build_views`` can be used to check
the equivalence against ``common.build_views`` on a small example.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

import common as C  # noqa: E402
import nn  # noqa: E402


def load_assets():
    """Load everything the submission scripts need, once."""
    df, genes = C.load_de()
    anchors = C.anchor_drugs(df)
    cells, drugs, cid, did = C.cat_indices(df)
    fp = C.drug_fingerprint_pca(df, n_comp=128)
    d2s, vocab = nn.build_smiles(df)
    idmap = pd.read_csv(C.data_path("id_map.csv"))
    template = pd.read_csv(C.data_path("sample_submission.csv"), nrows=0)
    gene_cols = list(template.columns[1:])
    drug_info = df[["drug", "SMILES", "sm_lincs_id"]].drop_duplicates("drug").set_index("drug")
    return dict(df=df, genes=genes, anchors=anchors, cells=cells, drugs=drugs,
                cid=cid, did=did, fp=fp, d2s=d2s, vocab=vocab, idmap=idmap,
                gene_cols=gene_cols, drug_info=drug_info)


def _mean(rows, Y, G):
    return Y[rows].mean(0) if len(rows) else np.zeros(G, dtype=np.float32)


def build_target_context(assets, target, n_dsig=64, n_csig=64):
    """Return the full context for one target cell type (B or Myeloid)."""
    df, genes = assets["df"], assets["genes"]
    anchors = set(assets["anchors"])
    idmap = assets["idmap"]
    Y = df[genes].to_numpy(np.float32)
    G = len(genes)

    test = idmap[idmap.cell_type == target].reset_index(drop=True)
    te_drugs = test["sm_name"].tolist()

    tr = df.index[df.cell_type.isin(C.FULL_CELLS) |
                  ((df.cell_type == target) & (df.drug.isin(anchors)))].to_numpy()
    tr_cells = df.loc[tr, "cell_type"].to_numpy()
    tr_drugs = df.loc[tr, "drug"].to_numpy()

    # visible (cell, drug) -> row: only training rows can contribute.
    lookup = {(c, d): int(i) for c, d, i in zip(tr_cells, tr_drugs, tr)}

    def drug_sig(cell, drug):
        src = [o for o in C.FULL_CELLS if o != cell]
        rows = [lookup[(o, drug)] for o in src if (o, drug) in lookup]
        return _mean(rows, Y, G)

    anchor_list = list(anchors)

    def cell_sig(cell, exclude_drug=None):
        mask = (tr_cells == cell) & np.isin(tr_drugs, anchor_list)
        if exclude_drug is not None:
            mask &= (tr_drugs != exclude_drug)
        return _mean(tr[mask], Y, G)

    # --- signatures ---------------------------------------------------------
    Dtr = np.stack([drug_sig(c, d) for c, d in zip(tr_cells, tr_drugs)]).astype(np.float32)
    Dte = np.stack([drug_sig(target, d) for d in te_drugs]).astype(np.float32)
    Ctr = np.stack([cell_sig(c, exclude_drug=d) for c, d in zip(tr_cells, tr_drugs)]).astype(np.float32)
    Cte = np.stack([cell_sig(target, exclude_drug=d) for d in te_drugs]).astype(np.float32)

    # --- fold-local PCA, fitted on training rows only ----------------------
    kd = min(n_dsig, Dtr.shape[0] - 1, Dtr.shape[1])
    pd_ = PCA(n_components=kd, random_state=C.SEED).fit(Dtr)
    Dtr_p, Dte_p = pd_.transform(Dtr), pd_.transform(Dte)

    src_cells = sorted(set(tr_cells))
    cs_all = np.vstack([np.stack([cell_sig(c) for c in src_cells]),
                        cell_sig(target).reshape(1, -1)])
    kc = min(n_csig, cs_all.shape[0] - 1)
    pc_ = PCA(n_components=max(kc, 1), random_state=C.SEED).fit(cs_all) if cs_all.shape[0] > 1 else None
    if pc_ is not None:
        Ctr_p, Cte_p = pc_.transform(Ctr), pc_.transform(Cte)
    else:
        Ctr_p = np.zeros((len(tr), 1), np.float32)
        Cte_p = np.zeros((len(te_drugs), 1), np.float32)

    fp_lookup = assets["fp"].set_index("drug")
    Ftr = fp_lookup.loc[tr_drugs].to_numpy(np.float32)
    Fte = fp_lookup.loc[te_drugs].to_numpy(np.float32)

    def compact(F, D, Cc):
        scale = np.stack([np.abs(D).mean(1), np.abs(Cc).mean(1)], 1)
        return np.concatenate([F, D, Cc, scale], 1).astype(np.float32)

    view = {
        "Xc_train": compact(Ftr, Dtr_p, Ctr_p),
        "Xc_test": compact(Fte, Dte_p, Cte_p),
        "Dsig_train": Dtr, "Dsig_test": Dte,
        "Csig_train": Ctr, "Csig_test": Cte,
        "y_train": Y[tr], "y_test": np.zeros((len(te_drugs), G), np.float32),
        "anchor_mask_train": (tr_cells == target) & np.isin(tr_drugs, list(anchors)),
        "anchor_mask_test": np.zeros(len(te_drugs), bool),
        "rows_train": list(zip(tr_cells, tr_drugs)),
        "rows_test": [(target, d) for d in te_drugs],
        "genes": genes,
    }

    cid, did = assets["cid"], assets["did"]
    x_tr = np.array([[cid[c], did[d]] for c, d in zip(tr_cells, tr_drugs)], dtype=np.int64)
    x_te = np.array([[cid[target], did[d]] for d in te_drugs], dtype=np.int64)
    sym_tr = np.stack([assets["d2s"][d] for d in tr_drugs])
    sym_te = np.stack([assets["d2s"][d] for d in te_drugs])

    return dict(view=view, ytr=view["y_train"], target=target,
                ids=test["id"].to_numpy(), te_drugs=te_drugs,
                x_tr=x_tr, x_te=x_te, sym_tr=sym_tr, sym_te=sym_te,
                lo=view["y_train"].min(0), hi=view["y_train"].max(0),
                train_idx=tr, cells=assets["cells"], drugs=assets["drugs"],
                cid=cid, did=did, d2s=assets["d2s"], vocab=assets["vocab"])


def assemble_submission(assets, parts, out_path):
    """parts: list of (ids, pred[n_rows, n_genes]) in any order."""
    idmap, genes, gene_cols = assets["idmap"], assets["genes"], assets["gene_cols"]
    pos = {k: i for i, k in enumerate(idmap["id"])}
    vals = np.zeros((len(idmap), len(genes)), np.float32)
    for ids, pred in parts:
        for k, row in zip(ids, pred):
            vals[pos[k]] = row
    out = pd.concat([idmap[["id"]].reset_index(drop=True),
                     pd.DataFrame(vals, columns=genes)], axis=1)[["id"] + gene_cols]
    assert out.shape == (255, 18212), out.shape
    assert list(out.columns) == ["id"] + gene_cols
    assert np.isfinite(out[gene_cols].to_numpy()).all()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.to_csv(out_path, index=False)
    print("wrote", out_path, out.shape)
    return out_path


def predict_both(assets, predict_one):
    """Run ``predict_one(ctx)`` for B and Myeloid and collect outputs."""
    parts = []
    for target in ["B cells", "Myeloid cells"]:
        ctx = build_target_context(assets, target)
        yp = np.asarray(predict_one(ctx), np.float32)
        yp = np.clip(yp, ctx["lo"], ctx["hi"])
        parts.append((ctx["ids"], yp))
        print(f"  {target:14s} pred mean|.|={np.abs(yp).mean():.3f}", flush=True)
    return parts


# --------------------------------------------------------------------------- #
# optional equivalence check against the (slow) reference implementation
# --------------------------------------------------------------------------- #
def verify_against_build_views(assets, target="B cells", n_rows=25, tol=1e-4):
    """Compare the fast context with ``common.build_views`` on the same fold."""
    df, genes, anchors = assets["df"], assets["genes"], assets["anchors"]
    test = assets["idmap"][assets["idmap"].cell_type == target].reset_index(drop=True)
    extra = pd.DataFrame(index=range(len(df), len(df) + len(test)), columns=df.columns)
    extra["cell_type"] = target
    extra["drug"] = test["sm_name"].to_numpy()
    extra["SMILES"] = [assets["drug_info"].at[d, "SMILES"] for d in test["sm_name"]]
    extra["sm_lincs_id"] = [assets["drug_info"].at[d, "sm_lincs_id"] for d in test["sm_name"]]
    extra["control"] = False
    extra[genes] = 0.0
    extra = extra.astype({c: df[c].dtype for c in df.columns if c not in genes})
    extra[genes] = extra[genes].astype(np.float32)
    da = pd.concat([df, extra])

    tr = np.arange(len(df))[df.cell_type.isin(C.FULL_CELLS) |
                            ((df.cell_type == target) & (df.drug.isin(set(anchors))))]
    te = extra.index.to_numpy()
    fold = C.Fold(name="check", protocol="test", target_cell=target,
                  anchors=list(anchors), ref_cells=list(C.FULL_CELLS),
                  train_idx=tr, test_idx=te, test_drugs=test["sm_name"].tolist())
    ref = C.build_views(da, genes, fold, assets["fp"])
    fast = build_target_context(assets, target)["view"]
    checks = {}
    for key in ["Xc_train", "Xc_test", "Dsig_train", "Dsig_test", "Csig_train", "Csig_test"]:
        a, b = ref[key], fast[key]
        checks[key] = dict(shape_ref=list(a.shape), shape_fast=list(b.shape),
                           allclose=bool(a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=1e-4)),
                           max_abs=float(np.max(np.abs(a - b))) if a.shape == b.shape else None)
    return checks


if __name__ == "__main__":
    import json
    assets = load_assets()
    print(json.dumps(verify_against_build_views(assets), indent=2))
