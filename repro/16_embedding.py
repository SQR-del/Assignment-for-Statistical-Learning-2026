"""#16 / tuned (cell, drug) embedding MLP.

Reference submission: ``results/submission_round2.csv`` (Kaggle 0.762 / 0.568).
The model is the round-2 winner: cell/drug embeddings -> MLP -> TruncatedSVD(25)
target latent, trained with the row-wise RMSE loss and a median over 5 seeds.

    python repro/16_embedding.py
    python repro/16_embedding.py --out results/repro/submission_round2.csv
    python repro/16_embedding.py --legacy      # emb=64, rank 25, lr=1e-3
"""
from __future__ import annotations

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))

import numpy as np

import nn
from target_context import load_assets, predict_both, assemble_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--n-dim", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20251001)
    ap.add_argument("--legacy", action="store_true",
                    help="old round-1 config: emb=64, hidden=(256,1024), lr=1e-3")
    ap.add_argument("--out", default="results/repro/submission_round2.csv")
    args = ap.parse_args()

    if args.legacy:
        args.emb, args.hidden, args.lr = 64, [256, 1024], 1e-3

    assets = load_assets()

    def predict_one(ctx):
        preds = [
            nn.train_embed(ctx["x_tr"], ctx["ytr"], ctx["x_te"],
                           len(ctx["cells"]), len(ctx["drugs"]),
                           emb=args.emb, hidden=tuple(args.hidden), n_dim=args.n_dim,
                           epochs=args.epochs, lr=args.lr, wd=args.wd, drop=args.drop,
                           seed=args.seed + r)
            for r in range(args.reps)
        ]
        return np.median(np.stack(preds), 0)

    print(f"embedding: emb={args.emb} hidden={tuple(args.hidden)} rank={args.n_dim} "
          f"epochs={args.epochs} lr={args.lr} wd={args.wd} reps={args.reps}")
    parts = predict_both(assets, predict_one)
    assemble_submission(assets, parts, args.out)


if __name__ == "__main__":
    main()
