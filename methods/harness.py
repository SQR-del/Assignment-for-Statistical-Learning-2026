"""Shared validation / export harness for the per-method folders.

A method is one callable
    predict(ctx, **params) -> array [n_test, n_genes]
where `ctx` is a fold context dict containing (see `Bench.contexts`):
    view    : feature dict from common.build_views (Xc/Dsig/Csig/y_train/...)
    ytr     : the target matrix of the training fold
    split   : integer ids & SMILES matrices for embedding models
    fold    : the common.Fold (protocol, target_cell, indices)
    d2s/vocab, cells/drugs/cid/did, anchors, df, genes, lo, hi

`run.py` in each folder calls  main(predict, __file__, params)  which evaluates on
the fixed leak-free protocols and writes scores.json + rows.csv here.

    python run.py                 # P1 + P2 internal validation -> scores.json, rows.csv
    python run.py --protocols P1
    python run.py --export        # train on all data -> results/submission_<name>.csv
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))

import common as C  # noqa: E402
import nn  # noqa: E402


class Bench:
    """Loads data once and yields fold contexts for the requested protocols."""

    def __init__(self, protocols=("P1", "P2")):
        self.protocols = protocols
        self.df, self.genes = C.load_de()
        self.anchors = C.anchor_drugs(self.df)
        self.cells, self.drugs, self.cid, self.did = C.cat_indices(self.df)
        self.fp = C.drug_fingerprint_pca(self.df, n_comp=128)
        self.d2s, self.vocab = nn.build_smiles(self.df)

    def _split(self, fold):
        return dict(
            x_tr=np.array([[self.cid[self.df.at[i, "cell_type"]], self.did[self.df.at[i, "drug"]]]
                           for i in fold.train_idx], dtype=np.int64),
            x_te=np.array([[self.cid[self.df.at[i, "cell_type"]], self.did[self.df.at[i, "drug"]]]
                           for i in fold.test_idx], dtype=np.int64),
            sym_tr=np.stack([self.d2s[self.df.at[i, "drug"]] for i in fold.train_idx]),
            sym_te=np.stack([self.d2s[self.df.at[i, "drug"]] for i in fold.test_idx]),
        )

    def contexts(self):
        for proto in self.protocols:
            for fold in C.make_folds(self.df, self.genes, self.anchors, proto):
                view = C.build_views(self.df, self.genes, fold, self.fp)
                yield dict(view=view, ytr=view["y_train"], fold=fold, df=self.df, genes=self.genes,
                           anchors=self.anchors, cells=self.cells, drugs=self.drugs,
                           cid=self.cid, did=self.did, d2s=self.d2s, vocab=self.vocab,
                           split=self._split(fold),
                           tv=C.train_gene_variance(view["y_train"]),
                           lo=view["y_train"].min(0), hi=view["y_train"].max(0))


def evaluate(predict, params, protocols, out_dir):
    bench = Bench(protocols)
    rows = []
    t0 = time.time()
    for k, ctx in enumerate(bench.contexts()):
        yp = np.asarray(predict(ctx, **params), np.float32)
        assert yp.shape == ctx["view"]["y_test"].shape, f"bad shape {yp.shape}"
        yp = np.clip(yp, ctx["lo"], ctx["hi"])
        m, _ = C.compute_metrics(ctx["view"]["y_test"], yp, ctx["genes"], train_var=ctx["tv"])
        rows.append(dict(protocol=ctx["fold"].protocol, fold=ctx["fold"].target_cell, model=ctx["fold"].name,
                         mrrmse=m["mrrmse"], mse=m["mse"], r2=m["r2"], pearson=m["pearson_mean"]))
        print(f"  {ctx['fold'].name:28s} mrrmse={m['mrrmse']:.4f} r2={m['r2']:+.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out_dir, "rows.csv"), index=False)
    scores = {}
    for proto in protocols:
        sub = R[R.protocol == proto]
        if len(sub):
            scores[proto] = {k: round(float(v), 4) for k, v in
                             sub[["mrrmse", "mse", "r2", "pearson"]].mean().items()}
    scores["params"] = params
    json.dump(scores, open(os.path.join(out_dir, "scores.json"), "w"), indent=2)
    print("\n=== internal validation (mean over folds) ===\n",
          R.groupby("protocol")[["mrrmse", "mse", "r2", "pearson"]].mean().round(4).to_string())
    return scores


def export(predict, params, name):
    """Train on all available rows and write results/submission_<name>.csv.

    Builds a real target view (test rows are scored by the id_map pairs) so both
    view-based and ctx-based methods work.
    """
    import importlib
    predict_mod = importlib.import_module("predict")
    bench = Bench(("P1",))
    df, genes = bench.df, bench.genes
    anchors = bench.anchors
    idmap = pd.read_csv(C.data_path("id_map.csv"))
    template = pd.read_csv(C.data_path("sample_submission.csv"), nrows=0)
    gene_cols = list(template.columns[1:])
    vals = np.zeros((len(idmap), len(genes)), np.float32)
    for target in ["B cells", "Myeloid cells"]:
        tc = predict_mod.build_target_context(df, genes, anchors, target, idmap, bench.fp, bench.cid, bench.did)
        tr = df.index[df.cell_type.isin(C.FULL_CELLS) |
                      ((df.cell_type == target) & (df.drug.isin(set(anchors))))].tolist()
        fold = C.Fold(name=f"test_{target}", protocol="test", target_cell=target,
                      anchors=list(anchors), ref_cells=[c for c in C.FULL_CELLS if c != target],
                      train_idx=np.array(tr), test_idx=np.array(tc["te_ids"]), test_drugs=tc["te_drugs"])
        sym_tr = np.stack([bench.d2s[df.at[i, "drug"]] for i in tr])
        sym_te = np.stack([bench.d2s[d] for d in tc["te_drugs"]])
        ctx = dict(view=tc["view"], ytr=tc["ytr"], fold=fold, df=df, genes=genes, anchors=anchors,
                   cells=bench.cells, drugs=bench.drugs, cid=bench.cid, did=bench.did,
                   d2s=bench.d2s, vocab=bench.vocab, target=target, te_ids=tc["te_ids"],
                   te_drugs=tc["te_drugs"], lo=tc["ytr"].min(0), hi=tc["ytr"].max(0),
                   tv=C.train_gene_variance(tc["ytr"]),
                   split=dict(x_tr=tc["x_tr"], x_te=tc["x_te"], sym_tr=sym_tr, sym_te=sym_te))
        yp = np.asarray(predict(ctx, **params), np.float32)
        vals[tc["te_ids"]] = np.clip(yp, ctx["lo"], ctx["hi"])
    out = pd.concat([idmap[["id"]].reset_index(drop=True),
                     pd.DataFrame(vals, columns=genes)], axis=1)[["id"] + gene_cols]
    assert out.shape == (255, 18212) and np.isfinite(out[gene_cols].to_numpy()).all()
    path = C.results_path(f"submission_{name}.csv")
    out.to_csv(path, index=False)
    print("wrote", path, out.shape)
    return path


def main(predict, file, params=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["P1", "P2"])
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    params = params or {}
    folder = os.path.dirname(os.path.abspath(file))
    name = os.path.basename(folder)
    if args.export:
        export(predict, params, name)
    else:
        evaluate(predict, params, args.protocols, folder)


if __name__ == "__main__":
    print("This is a library; run one of the methods/<NN_name>/run.py scripts.")
