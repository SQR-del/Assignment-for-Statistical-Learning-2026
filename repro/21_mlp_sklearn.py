"""#21 / sklearn MLP on the compact features.

Reference submission: ``results/submission_mlp_sklearn.csv`` (0.893 / 0.659).
Tuned config from nested CV: hidden=(256,128), alpha=0.01, early stopping.

    python repro/21_mlp_sklearn.py
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
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20251001)
    ap.add_argument("--out", default="results/repro/submission_mlp_sklearn.csv")
    args = ap.parse_args()

    assets = load_assets()
    print(f"mlp_sklearn: hidden={tuple(args.hidden)} alpha={args.alpha}")
    parts = predict_both(assets, lambda ctx: M.m_mlp_sklearn(
        ctx["view"], hidden=tuple(args.hidden), alpha=args.alpha,
        max_iter=args.max_iter, seed=args.seed))
    assemble_submission(assets, parts, args.out)


if __name__ == "__main__":
    main()
