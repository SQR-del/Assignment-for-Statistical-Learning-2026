"""#20 / PCA on the target space + Ridge.

Reference submission: ``results/submission_pca_ridge.csv`` (1.028 / 0.732).
The classic compact-feature model: standardise the 199-dim feature matrix,
PCA the per-gene standardised target to 128 components, fit Ridge(alpha=10).

    python repro/20_pca_ridge.py
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
    ap.add_argument("--n-comp", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--out", default="results/repro/submission_pca_ridge.csv")
    args = ap.parse_args()

    assets = load_assets()
    print(f"pca_ridge: n_comp={args.n_comp} alpha={args.alpha}")
    parts = predict_both(assets, lambda ctx: M.m_pca_ridge(
        ctx["view"], n_comp=args.n_comp, alpha=args.alpha))
    assemble_submission(assets, parts, args.out)


if __name__ == "__main__":
    main()
