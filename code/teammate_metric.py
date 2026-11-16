"""Teammate-compatible model evaluation: A/A15 + V1/V1w + objective.

This mirrors the metric used on the group repository's ``main`` branch
(``methods/harness.py`` and ``code/validate.py`` there):

    A   : for each T/NK cell type, hide all its non-anchor rows except controls,
          predict its official test compounds; A15 = A without CD8+.
    V1  : B / Myeloid leave-one-anchor-out (30 folds).
    V1w : only the anchors that are weak in the sources (row RMS of the source
          mean below the 90th percentile of the official test compounds).
    objective = mean(V1w / 1.1625, A15 / 0.9894)
                (KRR v1 = 1.000; LOWER IS BETTER)

Unlike ``common.build_views`` on the old P1/P2 protocols, this module keeps the
12 positive-control rows.  The metric's V1w weak/strong split uses only
NK / CD4+ / Treg (as on the main branch, because CD8+ lacks two anchors); the
models themselves may use all four complete cell types as signature sources,
which matches this repository's P1 feature design.  The feature builder below
is a vectorised equivalent of ``common.build_views`` for these folds;
``verify_views`` can check one fold against the reference implementation.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import common as C  # noqa: E402
import nn  # noqa: E402

TNK_NAMES = ("NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells")
# V1w classification on the main branch uses these three sources (CD8+ has two
# missing anchors).  Model inputs may use all four complete cells.
WEAK_SOURCES = ("NK cells", "T cells CD4+", "T regulatory cells")
FEATURE_SOURCES = tuple(C.FULL_CELLS)
SPARSE_NAMES = ("B cells", "Myeloid cells")
REF_KRR1 = dict(V1w=1.1625, A15=0.9894)


def load_assets():
    """Load the full table (controls included) and shared features."""
    df, genes = C.load_de(drop_controls=False)
    anchors = C.anchor_drugs(df)          # 15 non-control anchors
    cells, drugs, cid, did = C.cat_indices(df)
    fp = C.drug_fingerprint_pca(df, n_comp=128)
    d2s, vocab = nn.build_smiles(df)
    idmap = pd.read_csv(C.data_path("id_map.csv"))
    return dict(df=df, genes=genes, anchors=anchors, cells=cells, drugs=drugs,
                cid=cid, did=did, fp=fp, fp_lookup=fp.set_index("drug"),
                d2s=d2s, vocab=vocab, idmap=idmap)


def make_folds(assets):
    """Return the 34 teammate folds as ``common.Fold`` objects."""
    df, genes = assets["df"], assets["genes"]
    anchors = list(assets["anchors"])
    Y = df[genes].to_numpy(np.float64)
    row_of = {(c, d): i for i, (c, d) in enumerate(zip(df.cell_type, df.drug))}
    qoff = set(assets["idmap"].sm_name)
    src = [c for c in WEAK_SOURCES if c in set(df.cell_type)]

    # weak-anchor cut: 90th percentile of the source-mean row RMS over official
    # test compounds (same definition as the main-branch harness).
    src_rms = []
    for d in sorted(qoff):
        rows = [row_of[(c, d)] for c in src if (c, d) in row_of]
        if rows:
            src_rms.append(float(C.rms(Y[rows].mean(0))))
    cut = float(np.quantile(src_rms, 0.9))

    folds = []
    for o in TNK_NAMES:
        if o not in set(df.cell_type):
            continue
        hid = np.flatnonzero((df.cell_type == o).to_numpy()
                             & (~df.control.to_numpy())
                             & (~df.drug.isin(anchors).to_numpy()))
        qry = np.array([i for i in hid if df.at[i, "drug"] in qoff], dtype=int)
        train = np.setdiff1d(np.arange(len(df)), hid)
        folds.append(C.Fold(
            name=o, protocol="A", target_cell=o, anchors=anchors,
            ref_cells=[c for c in df.cell_type.unique() if c != o],
            train_idx=train, test_idx=qry,
            test_drugs=sorted(df.loc[qry, "drug"].unique())))

    for t in SPARSE_NAMES:
        if t not in set(df.cell_type):
            continue
        for d in anchors:
            i = row_of[(t, d)]
            rows = [row_of[(c, d)] for c in src if (c, d) in row_of]
            weak = float(C.rms(Y[rows].mean(0))) < cut
            name = f"{t}|{d}" + ("|weak" if weak else "")
            folds.append(C.Fold(
                name=name, protocol="V1", target_cell=t, anchors=anchors,
                ref_cells=[c for c in df.cell_type.unique() if c != t],
                train_idx=np.setdiff1d(np.arange(len(df)), [i]),
                test_idx=np.array([i]), test_drugs=[d]))
    return folds


def build_view_fast(df, genes, fold, fp_pca, source_cells, n_dsig=64, n_csig=64,
                    fp_lookup=None):
    """Vectorised equivalent of ``common.build_views`` (same columns/order)."""
    Y = df[genes].to_numpy(np.float32)
    G = len(genes)
    anchors = set(fold.anchors)
    anchor_list = list(anchors)
    sources = [c for c in source_cells]
    train_idx = np.asarray(fold.train_idx)
    test_idx = np.asarray(fold.test_idx)
    tr_cells = df.loc[train_idx, "cell_type"].to_numpy()
    tr_drugs = df.loc[train_idx, "drug"].to_numpy()
    te_cells = df.loc[test_idx, "cell_type"].to_numpy()
    te_drugs = df.loc[test_idx, "drug"].to_numpy()

    lookup = {(c, d): int(i) for c, d, i in zip(tr_cells, tr_drugs, train_idx)}

    def mean_rows(rows):
        return Y[rows].mean(0) if len(rows) else np.zeros(G, np.float32)

    def drug_sig(cell, drug):
        rows = [lookup[(o, drug)] for o in sources if o != cell and (o, drug) in lookup]
        return mean_rows(rows)

    def cell_sig(cell, exclude_drug=None):
        mask = (tr_cells == cell) & np.isin(tr_drugs, anchor_list)
        if exclude_drug is not None:
            mask &= (tr_drugs != exclude_drug)
        return mean_rows(train_idx[mask])

    Dtr = np.stack([drug_sig(c, d) for c, d in zip(tr_cells, tr_drugs)]).astype(np.float32)
    Dte = np.stack([drug_sig(c, d) for c, d in zip(te_cells, te_drugs)]).astype(np.float32)
    Ctr = np.stack([cell_sig(c, d) for c, d in zip(tr_cells, tr_drugs)]).astype(np.float32)
    Cte = np.stack([cell_sig(c, d) for c, d in zip(te_cells, te_drugs)]).astype(np.float32)

    kd = min(n_dsig, Dtr.shape[0] - 1, Dtr.shape[1])
    pd_ = PCA(n_components=kd, random_state=C.SEED).fit(Dtr)
    Dtr_p, Dte_p = pd_.transform(Dtr), pd_.transform(Dte)

    src_cells = sorted(set(tr_cells))
    cs_all = np.vstack([np.stack([cell_sig(c) for c in src_cells]),
                        cell_sig(fold.target_cell).reshape(1, -1)])
    kc = min(n_csig, cs_all.shape[0] - 1)
    pc_ = PCA(n_components=max(kc, 1), random_state=C.SEED).fit(cs_all) if cs_all.shape[0] > 1 else None
    if pc_ is not None:
        Ctr_p, Cte_p = pc_.transform(Ctr), pc_.transform(Cte)
    else:
        Ctr_p = np.zeros((len(train_idx), 1), np.float32)
        Cte_p = np.zeros((len(test_idx), 1), np.float32)

    fp_lookup = fp_lookup if fp_lookup is not None else fp_pca.set_index("drug")
    Ftr = fp_lookup.loc[tr_drugs].to_numpy(np.float32)
    Fte = fp_lookup.loc[te_drugs].to_numpy(np.float32)

    def compact(F, D, Cc):
        scale = np.stack([np.abs(D).mean(1), np.abs(Cc).mean(1)], 1)
        return np.concatenate([F, D, Cc, scale], 1).astype(np.float32)

    return {
        "Xc_train": compact(Ftr, Dtr_p, Ctr_p),
        "Xc_test": compact(Fte, Dte_p, Cte_p),
        "Dsig_train": Dtr, "Dsig_test": Dte,
        "Csig_train": Ctr, "Csig_test": Cte,
        "y_train": Y[train_idx], "y_test": Y[test_idx],
        "anchor_mask_train": (tr_cells == fold.target_cell) & np.isin(tr_drugs, list(anchors)),
        "anchor_mask_test": np.zeros(len(test_idx), bool),
        "rows_train": list(zip(tr_cells, tr_drugs)),
        "rows_test": list(zip(te_cells, te_drugs)),
        "genes": genes,
    }


def build_context(assets, fold):
    """Context with the same keys as ``methods.harness.Bench.contexts``."""
    df = assets["df"]
    view = build_view_fast(df, assets["genes"], fold, assets["fp"], FEATURE_SOURCES,
                           fp_lookup=assets.get("fp_lookup"))
    cid, did, d2s = assets["cid"], assets["did"], assets["d2s"]
    tr = np.asarray(fold.train_idx)
    te = np.asarray(fold.test_idx)
    tr_cells = df.loc[tr, "cell_type"].to_numpy()
    tr_drugs = df.loc[tr, "drug"].to_numpy()
    te_cells = df.loc[te, "cell_type"].to_numpy()
    te_drugs = df.loc[te, "drug"].to_numpy()
    split = dict(
        x_tr=np.array([[cid[c], did[d]] for c, d in zip(tr_cells, tr_drugs)], dtype=np.int64),
        x_te=np.array([[cid[c], did[d]] for c, d in zip(te_cells, te_drugs)], dtype=np.int64),
        sym_tr=np.stack([d2s[d] for d in tr_drugs]),
        sym_te=np.stack([d2s[d] for d in te_drugs]),
    )
    return dict(view=view, ytr=view["y_train"], fold=fold, df=df,
                genes=assets["genes"], anchors=assets["anchors"],
                cells=assets["cells"], drugs=assets["drugs"],
                cid=cid, did=did, d2s=d2s, vocab=assets["vocab"], split=split,
                tv=C.train_gene_variance(view["y_train"]),
                lo=view["y_train"].min(0), hi=view["y_train"].max(0))


def summarize(R):
    a = R[R.proto == "A"].groupby("name").err.mean()
    b = R[R.proto == "V1"].copy()
    b["target"] = b["name"].str.split("|").str[0]
    b["weak"] = b["name"].str.endswith("|weak")
    s = dict(A=a.mean(),
             A15=a.drop("T cells CD8+").mean() if "T cells CD8+" in a.index else a.mean(),
             V1=b.groupby("target").err.mean().mean(),
             V1w=b[b.weak].groupby("target").err.mean().mean())
    s["objective"] = (s["V1w"] / REF_KRR1["V1w"] + s["A15"] / REF_KRR1["A15"]) / 2
    s.update({f"A_{k}": v for k, v in a.items()})
    return {k: round(float(v), 4) for k, v in s.items()}


def _preserve_legacy(out_dir):
    """Keep old P1/P2 scores before the teammate metric overwrites scores.json."""
    sp = os.path.join(out_dir, "scores.json")
    lp = os.path.join(out_dir, "scores_p1p2.json")
    if os.path.exists(sp) and not os.path.exists(lp):
        try:
            d = json.load(open(sp, encoding="utf-8"))
            if "P1" in d or "P2" in d:
                shutil.copy2(sp, lp)
        except Exception:
            pass
    rp = os.path.join(out_dir, "rows.csv")
    lp = os.path.join(out_dir, "rows_p1p2.csv")
    if os.path.exists(rp) and not os.path.exists(lp):
        try:
            d = pd.read_csv(rp)
            if "protocol" in d.columns and d["protocol"].isin(["P1", "P2"]).any():
                shutil.copy2(rp, lp)
        except Exception:
            pass


def evaluate(predict, params, out_dir):
    """Evaluate one ``predict(ctx, **params)`` on the 34 teammate folds."""
    assets = load_assets()
    folds = make_folds(assets)
    recs = []
    t0 = time.time()
    for k, fold in enumerate(folds):
        ctx = build_context(assets, fold)
        yp = np.asarray(predict(ctx, **params), np.float32)
        yp = np.clip(yp, ctx["lo"], ctx["hi"])
        truth = ctx["view"]["y_test"]
        assert yp.shape == truth.shape, f"bad shape {yp.shape} vs {truth.shape}"
        recs += [dict(proto=fold.protocol, name=fold.name, row=int(i), err=float(e))
                 for i, e in zip(fold.test_idx, C.rms(yp - truth))]
        print(f"  fold {k + 1}/{len(folds)} {fold.protocol} {fold.name[:40]} "
              f"[t={time.time() - t0:.0f}s]", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    _preserve_legacy(out_dir)
    R = pd.DataFrame(recs)
    R.to_csv(os.path.join(out_dir, "rows.csv"), index=False)
    s = summarize(R)
    s["params"] = params
    json.dump(s, open(os.path.join(out_dir, "scores.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\n=== teammate metric (lower is better; KRR v1 = 1.000) ===")
    print(json.dumps(s, indent=2, ensure_ascii=False))
    return s


def evaluate_methods(specs, names, out_dir=None):
    """Evaluate several methods on the same 34 folds.

    ``specs[name]`` is a callable ``predict(ctx) -> array`` (params already bound).
    Returns (rows DataFrame, summary DataFrame indexed by method).
    """
    assets = load_assets()
    folds = make_folds(assets)
    recs = []
    t0 = time.time()
    for k, fold in enumerate(folds):
        ctx = build_context(assets, fold)
        for name in names:
            t1 = time.time()
            try:
                yp = np.asarray(specs[name](ctx), np.float32)
                yp = np.clip(yp, ctx["lo"], ctx["hi"])
                truth = ctx["view"]["y_test"]
                assert yp.shape == truth.shape
                recs += [dict(model=name, proto=fold.protocol, name=fold.name,
                              row=int(i), err=float(e),
                              seconds=round(time.time() - t1, 2))
                         for i, e in zip(fold.test_idx, C.rms(yp - truth))]
            except Exception as exc:
                recs.append(dict(model=name, proto=fold.protocol, name=fold.name,
                                 row=np.nan, err=np.nan, seconds=round(time.time() - t1, 2),
                                 error=repr(exc)))
                print(f"  {name} FAILED on {fold.name}: {exc!r}", flush=True)
        print(f"  fold {k + 1}/{len(folds)} {fold.protocol} {fold.name[:40]} "
              f"[t={time.time() - t0:.0f}s]", flush=True)

    R = pd.DataFrame(recs)
    ok = R[R.get("error").isna()] if "error" in R else R
    S = pd.DataFrame({m: summarize(g) for m, g in ok.groupby("model")}).T
    S = S.sort_values("objective")
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        R.to_csv(os.path.join(out_dir, "benchmark_teammate_raw.csv"), index=False)
        S.to_csv(os.path.join(out_dir, "benchmark_teammate_summary.csv"))
    return R, S


def verify_views(assets, kind="A", index=0, tol=1e-4):
    """Compare ``build_view_fast`` with ``common.build_views`` on one fold."""
    folds = [f for f in make_folds(assets) if f.protocol == kind]
    fold = folds[index]
    ref = C.build_views(assets["df"], assets["genes"], fold, assets["fp"],
                        source_cells=FEATURE_SOURCES)
    fast = build_context(assets, fold)["view"]
    out = {}
    for key in ["Xc_train", "Xc_test", "Dsig_train", "Dsig_test", "Csig_train", "Csig_test"]:
        a, b = ref[key], fast[key]
        out[key] = dict(shape_ref=list(a.shape), shape_fast=list(b.shape),
                        allclose=bool(a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=1e-4)),
                        max_abs=float(np.max(np.abs(a - b))) if a.shape == b.shape else None)
    return out


if __name__ == "__main__":
    print(json.dumps(verify_views(load_assets()), indent=2))
