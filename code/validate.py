"""Benchmark driver + sanity checks.

The default evaluation is now the teammate-compatible metric from the group
repository's `main` branch: 34 folds (A + V1), with A15 / V1w and the
pre-declared objective

    objective = mean(V1w / 1.1625, A15 / 0.9894)   (lower is better, KRR v1 = 1)

It writes `results/benchmark_teammate_raw.csv` and
`results/benchmark_teammate_summary.csv`.

The legacy P1/P2 benchmark is still available with `--metric p1p2`.

Usage:
    python code/validate.py                              # teammate metric, all methods
    python code/validate.py --methods ridge ensemble      # a subset
    python code/validate.py --metric p1p2 --protocols P1  # legacy benchmark
    python code/validate.py --sanity                     # leakage / correctness checks
"""
from __future__ import annotations

import os
import sys
import argparse
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C
import models as M
import nn
import teammate_metric as TM


def _ids(df, idx, cid, did):
    return np.array([[cid[df.at[i, "cell_type"]], did[df.at[i, "drug"]]] for i in idx], dtype=np.int64)


def method_matrix_specs(df, genes, folds, fp, cid, did, d2s, vocab, cells, drugs):
    """Build the (name -> (callable(ctx) -> pred, seed)) registry that needs ctx."""
    specs = {}
    for name, (fn, params) in M.CLASSIC_METHODS.items():
        specs[name] = (lambda ctx, fn=fn, params=params: fn(ctx["view"], **params), "view")

    def embed(ctx):
        s = ctx["split"]
        return nn.train_embed(s["x_tr"], ctx["ytr"], s["x_te"], len(cells), len(drugs),
                              n_dim=25, seed=C.SEED)

    def hybrid(ctx):
        from sklearn.decomposition import PCA
        s = ctx["split"]
        pd_ = PCA(n_components=64, random_state=C.SEED).fit(ctx["view"]["Dsig_train"])
        Xtr = pd_.transform(ctx["view"]["Dsig_train"]).astype(np.float32)
        Xte = pd_.transform(ctx["view"]["Dsig_test"]).astype(np.float32)
        return nn.train_hybrid(s["x_tr"], Xtr, ctx["ytr"], s["x_te"], Xte, len(cells), len(drugs),
                               n_dim=25, seed=C.SEED)

    def smiles(ctx):
        s = ctx["split"]
        return nn.train_smiles(s["x_tr"], s["sym_tr"], ctx["ytr"], s["x_te"], s["sym_te"],
                               vocab, len(cells), len(drugs), n_dim=25, seed=C.SEED)

    def full_mlp(ctx):
        s = ctx["split"]
        Xtr, Xte = M._scaled(ctx["view"])
        return nn.train_full_mlp(Xtr, Xte, ctx["ytr"], hidden=(512, 256), epochs=150, seed=C.SEED)

    specs.update({
        "embedding": (embed, "ctx"),
        "hybrid": (hybrid, "ctx"),
        "smiles_cnn": (smiles, "ctx"),
        "mlp_full": (full_mlp, "ctx"),
    })
    return specs


def run(protocols, method_names, save=True):
    df, genes = C.load_de()
    anchors = C.anchor_drugs(df)
    cells, drugs, cid, did = C.cat_indices(df)
    fp = C.drug_fingerprint_pca(df, n_comp=128)
    d2s, vocab = nn.build_smiles(df)
    records = []

    for proto in protocols:
        for fold in C.make_folds(df, genes, anchors, proto):
            view = C.build_views(df, genes, fold, fp)
            tv = C.train_gene_variance(view["y_train"])
            lo, hi = view["y_train"].min(0), view["y_train"].max(0)
            split = dict(x_tr=_ids(df, fold.train_idx, cid, did),
                         x_te=_ids(df, fold.test_idx, cid, did),
                         sym_tr=np.stack([d2s[df.at[i, "drug"]] for i in fold.train_idx]),
                         sym_te=np.stack([d2s[d] for d in fold.test_drugs]))
            # sym_te must follow test row order
            split["sym_te"] = np.stack([d2s[df.at[i, "drug"]] for i in fold.test_idx])
            ctx = dict(view=view, ytr=view["y_train"], split=split, fold=fold)
            specs = method_matrix_specs(df, genes, None, fp, cid, did, d2s, vocab, cells, drugs)
            for name in method_names:
                fn, kind = specs[name]
                t0 = time.time()
                try:
                    yp = np.asarray(fn(ctx), np.float32)
                    yp = np.clip(yp, lo, hi)
                    assert yp.shape == view["y_test"].shape
                    m, _ = C.compute_metrics(view["y_test"], yp, genes, train_var=tv)
                    records.append(dict(protocol=proto, fold=fold.target_cell, model=name,
                                        seconds=round(time.time() - t0, 1), **m))
                    print(f"  {proto} {fold.target_cell:20s} {name:16s} "
                          f"mrrmse={m['mrrmse']:.4f} r2={m['r2']:+.3f} ({time.time()-t0:.1f}s)", flush=True)
                except Exception:
                    print(f"  {proto} {fold.target_cell:20s} {name:16s} FAILED")
                    traceback.print_exc()
                    records.append(dict(protocol=proto, fold=fold.target_cell, model=name, error="fail"))

    res = pd.DataFrame(records)
    if save and len(res):
        res.to_csv(C.results_path("benchmark_raw.csv"), index=False)
        ok = res[res.get("error").isna()] if "error" in res else res
        s = ok.groupby(["protocol", "model"])[["mrrmse", "mse", "r2", "pearson_mean"]].mean()
        s = s.sort_values("mrrmse")
        s.to_csv(C.results_path("benchmark_summary.csv"))
        print("\n=== benchmark summary (mean over folds) ===\n", s.round(4).to_string())
    return res


def run_teammate(method_names):
    """Evaluate registered methods on the 34 teammate folds (A + V1)."""
    df, genes = C.load_de(drop_controls=False)
    anchors = C.anchor_drugs(df)
    cells, drugs, cid, did = C.cat_indices(df)
    fp = C.drug_fingerprint_pca(df, n_comp=128)
    d2s, vocab = nn.build_smiles(df)
    specs = method_matrix_specs(df, genes, None, fp, cid, did, d2s, vocab, cells, drugs)
    names = [n for n in method_names if n in specs]
    missing = [n for n in method_names if n not in specs]
    if missing:
        print("unknown methods:", missing)
    callables = {n: (lambda ctx, fn=specs[n][0]: fn(ctx)) for n in names}
    R, S = TM.evaluate_methods(callables, names, out_dir=C.RESULTS_DIR)
    print("\n=== teammate metric (lower is better; objective = mean(V1w/1.1625, A15/0.9894)) ===")
    print(S.round(4).to_string())
    return R, S


def sanity():
    """Correctness / leakage checks (see REPORT.md §3.2)."""
    df, genes = C.load_de()
    anchors = C.anchor_drugs(df)
    fp = C.drug_fingerprint_pca(df, n_comp=128)
    Y = df[genes].to_numpy(np.float32)
    out = []

    def chk(name, cond, detail=""):
        out.append(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
        print(out[-1])

    chk("anchor drug count == 15", len(anchors) == 15, f"({len(anchors)})")
    for proto in ["P1", "P2"]:
        for f in C.make_folds(df, genes, anchors, proto):
            chk(f"{proto} {f.target_cell} train/test disjoint",
                len(set(f.train_idx) & set(f.test_idx)) == 0)

    fold = C.make_folds(df, genes, anchors, "P1")[0]
    view = C.build_views(df, genes, fold, fp)
    c, d = view["rows_test"][0]
    manual = np.mean([Y[df.index[(df.cell_type == o) & (df.drug == d)]][0]
                      for o in [o for o in C.FULL_CELLS if o != c]], 0)
    chk("dsig excludes the target cell type", np.allclose(manual, view["Dsig_test"][0], atol=1e-4))

    # negative control: shuffled targets should carry no signal
    rng = np.random.default_rng(0)
    v2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in view.items()}
    v2["y_train"] = view["y_train"][rng.permutation(view["y_train"].shape[0])]
    yp = np.clip(M.m_ridge(v2), view["y_train"].min(0), view["y_train"].max(0))
    m, _ = C.compute_metrics(view["y_test"], yp, genes)
    chk("negative control (shuffled y) R2 <= 0.02", m["r2"] <= 0.02, f"r2={m['r2']:.4f}")
    m, _ = C.compute_metrics(view["y_test"], view["y_test"], genes)
    chk("perfect prediction -> R2 == 1", abs(m["r2"] - 1) < 1e-6)

    with open(C.results_path("sanity_checks.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["teammate", "p1p2"], default="teammate",
                    help="teammate (default): A15/V1w objective; p1p2: legacy benchmark")
    ap.add_argument("--protocols", nargs="+", default=["P1", "P2"])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    if args.sanity:
        sanity(); return
    names = args.methods or list(M.CLASSIC_METHODS) + ["embedding", "hybrid", "smiles_cnn", "mlp_full"]
    if args.metric == "teammate":
        run_teammate(names)
    else:
        run(args.protocols, names)


if __name__ == "__main__":
    main()
