"""Compare a regenerated submission with an archived one.

The Kaggle private labels are not public, so the archived CSV is the proxy for
"the same predictions".  We report global correlation, mean row-wise RMSE and
the median per-row correlation.

    python repro/check_submissions.py results/repro/submission_pca_ridge.csv results/submission_pca_ridge.csv
    python repro/check_submissions.py --all
"""
from __future__ import annotations

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))

import numpy as np
import pandas as pd


PAIRS = [
    ("results/repro/submission_round2.csv", "results/submission_round2.csv", "#16 embedding"),
    ("results/repro/submission_pca_ridge.csv", "results/submission_pca_ridge.csv", "#20 pca_ridge"),
    ("results/repro/submission_mlp_sklearn.csv", "results/submission_mlp_sklearn.csv", "#21 mlp_sklearn"),
    ("results/repro/submission_mlp_wide.csv", "results/submission_mlp_wide.csv", "#22 mlp_wide"),
    ("results/repro/submission_ensemble.csv", "results/submission_ensemble.csv", "#23 ensemble"),
]


def compare(new_path, ref_path, by_cell=False):
    new = pd.read_csv(new_path)
    ref = pd.read_csv(ref_path)
    genes = [c for c in ref.columns if c != "id"]
    assert list(new["id"]) == list(ref["id"]), "id order differs"
    A = new[genes].to_numpy(np.float64)
    B = ref[genes].to_numpy(np.float64)
    resid = A - B
    row_rmse = np.sqrt(np.mean(resid ** 2, axis=1))
    row_corr = np.array([np.corrcoef(A[i], B[i])[0, 1] for i in range(len(A))])
    out = dict(
        global_corr=float(np.corrcoef(A.ravel(), B.ravel())[0, 1]),
        mean_row_rmse=float(row_rmse.mean()),
        max_abs_diff=float(np.abs(resid).max()),
        median_row_corr=float(np.median(row_corr)),
        mean_abs_new=float(np.abs(A).mean()),
        mean_abs_ref=float(np.abs(B).mean()),
    )
    if by_cell:
        idmap = pd.read_csv(os.path.join(ROOT, "data", "id_map.csv"))
        out["by_cell"] = {}
        for cell in ["B cells", "Myeloid cells"]:
            m = (idmap["cell_type"] == cell).to_numpy()
            out["by_cell"][cell] = dict(
                mean_row_rmse=float(row_rmse[m].mean()),
                median_row_corr=float(np.median(row_corr[m])),
                mean_abs_new=float(np.abs(A[m]).mean()),
                mean_abs_ref=float(np.abs(B[m]).mean()),
            )
    return out


def verdict(m):
    if m["global_corr"] > 0.999 and m["mean_row_rmse"] < 1e-3:
        return "EXACT"
    if m["global_corr"] > 0.99:
        return "NEAR"
    if m["global_corr"] > 0.95:
        return "CLOSE"
    return "DIFFERENT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new", nargs="?")
    ap.add_argument("ref", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--by-cell", action="store_true")
    args = ap.parse_args()

    pairs = [(a, b, c) for a, b, c in PAIRS if os.path.exists(os.path.join(ROOT, a))
             and os.path.exists(os.path.join(ROOT, b))] if args.all else []
    if args.all:
        if not pairs:
            print("no regenerated files found under results/repro/")
            return
        for a, b, label in pairs:
            m = compare(os.path.join(ROOT, a), os.path.join(ROOT, b), args.by_cell)
            print(f"{label:20s} {verdict(m):9s} corr={m['global_corr']:.6f} "
                  f"rowRMSE={m['mean_row_rmse']:.4f} "
                  f"mean|new|={m['mean_abs_new']:.3f} mean|ref|={m['mean_abs_ref']:.3f}")
            if args.by_cell:
                for cell, v in m["by_cell"].items():
                    print(f"    {cell:14s} rowRMSE={v['mean_row_rmse']:.4f} "
                          f"rowCorr={v['median_row_corr']:.4f}")
        return

    if not (args.new and args.ref):
        ap.error("give NEW and REF, or use --all")
    m = compare(args.new, args.ref, args.by_cell)
    print(f"{verdict(m):9s} corr={m['global_corr']:.6f} rowRMSE={m['mean_row_rmse']:.6f} "
          f"maxAbs={m['max_abs_diff']:.6f} medianRowCorr={m['median_row_corr']:.6f}")
    print(f"mean|new|={m['mean_abs_new']:.4f} mean|ref|={m['mean_abs_ref']:.4f}")
    if args.by_cell:
        for cell, v in m["by_cell"].items():
            print(f"{cell:14s} rowRMSE={v['mean_row_rmse']:.4f} "
                  f"rowCorr={v['median_row_corr']:.4f} "
                  f"mean|new|={v['mean_abs_new']:.3f} mean|ref|={v['mean_abs_ref']:.3f}")


if __name__ == "__main__":
    main()
