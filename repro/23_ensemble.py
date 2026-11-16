"""#23 / classic linear-kernel ensemble.

Reference submission: ``results/submission_ensemble.csv`` (0.885 / 0.645).
Equal-weight mean of five families on the compact features:
target-PCA Ridge + Ridge + PLS + KNN + linear kernel ridge on [dsig, csig].
Tuned config: ridge_alpha=100, knn_k=10, pls_n=10.

    python repro/23_ensemble.py
"""
from __future__ import annotations

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))

import models as M
from target_context import load_assets, predict_both, assemble_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge-alpha", type=float, default=100.0)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--pls-n", type=int, default=10)
    ap.add_argument("--n-comp", type=int, default=128)
    ap.add_argument("--out", default="results/repro/submission_ensemble.csv")
    args = ap.parse_args()

    assets = load_assets()
    print(f"ensemble: ridge_alpha={args.ridge_alpha} knn_k={args.knn_k} "
          f"pls_n={args.pls_n} n_comp={args.n_comp}")
    parts = predict_both(assets, lambda ctx: M.m_ensemble(
        ctx["view"], ridge_alpha=args.ridge_alpha, knn_k=args.knn_k,
        pls_n=args.pls_n, n_comp=args.n_comp))
    assemble_submission(assets, parts, args.out)


if __name__ == "__main__":
    main()
