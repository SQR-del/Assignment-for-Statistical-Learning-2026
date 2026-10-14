"""Build the Kaggle submission for the real test set (id_map.csv).

For each target cell type (B cells, Myeloid cells):
    train = 4 full cell types (all drugs) + the target's 15 anchor drugs
    test  = the id_map rows for that target
Features are built leakage-safely (dsig uses only the OTHER full cell types).

Default combines the round-2 winners with fixed weights (see README / REPORT):
    0.50 * embedding(ids, SVD rank 25) + 0.30 * SMILES-CNN + 0.20 * hybrid(ids+dsig)
Each component is averaged over several random seeds.

Usage:
    python code/predict.py                       # write results/submission.csv
    python code/predict.py --seeds 5 --embed-w 0.5 --smiles-w 0.3 --hybrid-w 0.2
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C
from models import m_ensemble
import nn


def build_target_context(df, genes, anchors, target, idmap, fp, cid, did):
    """Training context + test pairs for one Kaggle target cell type (leak-free)."""
    Y = df[genes].to_numpy(np.float32)
    anchors = set(anchors)
    tr = df.index[df.cell_type.isin(C.FULL_CELLS) |
                  ((df.cell_type == target) & (df.drug.isin(anchors)))].tolist()
    ids = idmap.index[idmap.cell_type == target].tolist()
    te_drugs = [idmap.at[i, "sm_name"] for i in ids]
    ytr = Y[tr]

    def drug_sig(cell, drug):
        vs = [Y[df.index[(df.cell_type == o) & (df.drug == drug)]][0]
              for o in C.FULL_CELLS if o != cell
              and len(df.index[(df.cell_type == o) & (df.drug == drug)])]
        return np.mean(vs, 0) if vs else np.zeros(len(genes), np.float32)

    Dtr = np.stack([drug_sig(df.at[i, "cell_type"], df.at[i, "drug"]) for i in tr]).astype(np.float32)
    Dte = np.stack([drug_sig(target, d) for d in te_drugs]).astype(np.float32)
    x_tr = np.array([[cid[df.at[i, "cell_type"]], did[df.at[i, "drug"]]] for i in tr], dtype=np.int64)
    x_te = np.array([[cid[target], did[d]] for d in te_drugs], dtype=np.int64)
    # compact view for the classic ensemble
    fp_lookup = fp.set_index("drug")
    Ftr = fp_lookup.loc[df.loc[tr, "drug"].values].to_numpy(np.float32)
    Fte = fp_lookup.loc[te_drugs].to_numpy(np.float32)
    pd_ = PCA(n_components=min(64, len(tr) - 1), random_state=C.SEED).fit(Dtr)
    def compact(F, D):
        return np.concatenate([F, pd_.transform(D)], 1).astype(np.float32)
    view = dict(Xc_train=compact(Ftr, Dtr), Xc_test=compact(Fte, Dte),
                Dsig_train=Dtr, Dsig_test=Dte, y_train=ytr,
                Csig_train=np.zeros((len(tr), 1), np.float32), Csig_test=np.zeros((len(te_drugs), 1), np.float32))
    return dict(view=view, ytr=ytr, x_tr=x_tr, x_te=x_te, Dtr=Dtr, Dte=Dte,
                te_ids=ids, te_drugs=te_drugs, target=target)


def component_predictions(ctx, df, genes, n_cells, n_drugs, d2s, vocab, cfg):
    """Return {name: (n_test, G) prediction} for the configured blend components."""
    x_tr, x_te, ytr = ctx["x_tr"], ctx["x_te"], ctx["ytr"]
    out = {}
    if cfg["embed"]:
        out["embed"] = np.median(np.stack([
            nn.train_embed(x_tr, ytr, x_te, n_cells, n_drugs, emb=cfg["emb"], hidden=cfg["hidden"],
                           n_dim=cfg["n_dim"], epochs=cfg["epochs"], lr=cfg["lr"], drop=cfg["drop"],
                           seed=C.SEED + r) for r in range(cfg["seeds_embed"])]), 0)
    if cfg["smiles"]:
        sym_tr = np.stack([d2s[df.at[i, "drug"]] for i in ctx["_tr_idx"]])
        sym_te = np.stack([d2s[d] for d in ctx["te_drugs"]])
        out["smiles"] = np.median(np.stack([
            nn.train_smiles(x_tr, sym_tr, ytr, x_te, sym_te, vocab, n_cells, n_drugs,
                            n_dim=cfg["n_dim"], seed=C.SEED + r)
            for r in range(cfg["seeds_other"])]), 0)
    if cfg["hybrid"]:
        pd_ = PCA(n_components=64, random_state=C.SEED).fit(ctx["Dtr"])
        Xtr = pd_.transform(ctx["Dtr"]).astype(np.float32)
        Xte = pd_.transform(ctx["Dte"]).astype(np.float32)
        out["hybrid"] = np.median(np.stack([
            nn.train_hybrid(x_tr, Xtr, ytr, x_te, Xte, n_cells, n_drugs, emb=cfg["emb"],
                            hidden=cfg["hidden"], n_dim=cfg["n_dim"], epochs=cfg["epochs"],
                            lr=cfg["lr"], drop=cfg["drop"], seed=C.SEED + r)
            for r in range(cfg["seeds_other"])]), 0)
    if cfg["ensemble"]:
        fn, params = cfg["ensemble_fn"], cfg["ensemble_params"]
        out["ensemble"] = np.asarray(fn(ctx["view"], **params), np.float32)
    return out


DEFAULT_CFG = dict(embed=True, smiles=True, hybrid=True, ensemble=False,
                   emb=128, hidden=(256, 512, 1024), n_dim=25, epochs=400, lr=3e-3, drop=0.2,
                   seeds_embed=7, seeds_other=3,
                   embed_w=0.50, smiles_w=0.30, hybrid_w=0.20, ensemble_w=0.0,
                   ensemble_fn=m_ensemble, ensemble_params=dict(ridge_alpha=100, knn_k=10, pls_n=10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--embed-w", type=float, default=0.50)
    ap.add_argument("--smiles-w", type=float, default=0.30)
    ap.add_argument("--hybrid-w", type=float, default=0.20)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    cfg["seeds_embed"] = args.seeds
    cfg["embed_w"], cfg["smiles_w"], cfg["hybrid_w"] = args.embed_w, args.smiles_w, args.hybrid_w

    df, genes = C.load_de()
    anchors = C.anchor_drugs(df)
    cells, drugs, cid, did = C.cat_indices(df)
    Y = df[genes].to_numpy(np.float32)
    fp = C.drug_fingerprint_pca(df, n_comp=128)
    d2s, vocab = nn.build_smiles(df)

    idmap = pd.read_csv(C.data_path("id_map.csv"))
    template = pd.read_csv(C.data_path("sample_submission.csv"), nrows=0)
    gene_cols = list(template.columns[1:])
    assert sorted(gene_cols) == sorted(genes), "gene set mismatch"

    vals = np.zeros((len(idmap), len(genes)), np.float32)
    for target in ["B cells", "Myeloid cells"]:
        t0 = time.time()
        ctx = build_target_context(df, genes, anchors, target, idmap, fp, cid, did)
        ctx["_tr_idx"] = df.index[df.cell_type.isin(C.FULL_CELLS) |
                                  ((df.cell_type == target) & (df.drug.isin(set(anchors))))].tolist()
        lo, hi = ctx["ytr"].min(0), ctx["ytr"].max(0)
        comp = component_predictions(ctx, df, genes, len(cells), len(drugs), d2s, vocab, cfg)
        weights = dict(embed=cfg["embed_w"], smiles=cfg["smiles_w"], hybrid=cfg["hybrid_w"])
        blend = sum(weights[k] * np.asarray(v, np.float32) for k, v in comp.items())
        blend = np.clip(blend, lo, hi).astype(np.float32)
        vals[ctx["te_ids"]] = blend
        msg = " ".join(f"|{k}|={np.abs(v).mean():.3f}" for k, v in comp.items())
        print(f"  {target:14s} {msg} ({time.time()-t0:.0f}s)", flush=True)

    out = pd.concat([idmap[["id"]].reset_index(drop=True),
                     pd.DataFrame(vals, columns=genes)], axis=1)[["id"] + gene_cols]
    assert out.shape == (255, 18212), out.shape
    assert list(out.columns) == ["id"] + gene_cols
    assert np.isfinite(out[gene_cols].to_numpy()).all()
    path = C.results_path(args.out)
    out.to_csv(path, index=False)
    C.results_path("submission.csv")  # ensure dir
    print("format OK:", out.shape, "->", path)


if __name__ == "__main__":
    main()
