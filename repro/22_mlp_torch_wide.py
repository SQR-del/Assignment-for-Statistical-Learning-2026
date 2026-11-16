"""#22 / wide PyTorch MLP on the full gene space.

Reference submission: ``results/submission_mlp_wide.csv`` (0.954 / 0.686).

IMPORTANT: the archived wide-MLP submission was produced by the *early* config
``hidden=(1024,512), 150 epochs, wd=1e-4``.  The later nested-CV selection
(100 epochs, wd=1e-3) is visibly worse on the real B/Myeloid test set and was
the source of the +0.112 private-score gap.  This script therefore defaults to
the early config; use ``--tuned`` for the nested-CV variant.

    python repro/22_mlp_torch_wide.py
    python repro/22_mlp_torch_wide.py --tuned
"""
from __future__ import annotations

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))

import numpy as np

import models as M
import nn
from target_context import load_assets, predict_both, assemble_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20251001)
    ap.add_argument("--tuned", action="store_true",
                    help="nested-CV config: 100 epochs, wd=1e-3 (worse on LB)")
    ap.add_argument("--out", default="results/repro/submission_mlp_wide.csv")
    args = ap.parse_args()

    if args.tuned:
        args.epochs, args.wd = 100, 1e-3

    def predict_one(ctx):
        Xtr, Xte = M._scaled(ctx["view"])
        preds = [
            nn.train_full_mlp(Xtr, Xte, ctx["ytr"], hidden=tuple(args.hidden),
                              epochs=args.epochs, lr=args.lr, wd=args.wd,
                              dropout=args.dropout, cosine=True, seed=args.seed + r)
            for r in range(args.reps)
        ]
        return np.median(np.stack(preds), 0) if args.reps > 1 else preds[0]

    print(f"mlp_torch_wide: hidden={tuple(args.hidden)} epochs={args.epochs} "
          f"lr={args.lr} wd={args.wd} reps={args.reps}")
    assets = load_assets()
    parts = predict_both(assets, predict_one)
    assemble_submission(assets, parts, args.out)


if __name__ == "__main__":
    main()
