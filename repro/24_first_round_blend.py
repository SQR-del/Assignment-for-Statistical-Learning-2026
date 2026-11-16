"""#24 / round-1 blend.

The round-1 main submission was

    0.6 * embedding(emb=64, SVD rank 25, 5-seed median) + 0.4 * ensemble

(``REPORT.md`` sections 16-17).  It is NOT the round-2 recipe
``0.5*embed + 0.3*SMILES + 0.2*hybrid`` implemented in ``code/predict.py``.

Recipes:
  first   (default) 0.60 * embed64 + 0.40 * ensemble
  improve            0.65 * embed64 + 0.35 * ensemble  (matches the archived
                     ``submission_improve_blend.csv`` to corr 0.997)

    python repro/24_first_round_blend.py --embed-reps 5
    python repro/24_first_round_blend.py --recipe improve
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
from target_context import load_assets, build_target_context, assemble_submission

RECIPES = {
    "first": dict(w_embed=0.60),
    "improve": dict(w_embed=0.65),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="first")
    ap.add_argument("--embed-reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20251001)
    ap.add_argument("--out", default="results/repro/submission_first_round_blend.csv")
    args = ap.parse_args()

    w_embed = RECIPES[args.recipe]["w_embed"]
    assets = load_assets()
    parts = []
    for target in ["B cells", "Myeloid cells"]:
        ctx = build_target_context(assets, target)
        emb = np.median(np.stack([
            nn.train_embed(ctx["x_tr"], ctx["ytr"], ctx["x_te"],
                           len(ctx["cells"]), len(ctx["drugs"]),
                           emb=64, hidden=(256, 1024), n_dim=25, epochs=400,
                           lr=1e-3, wd=1e-4, drop=0.2, seed=args.seed + r)
            for r in range(args.embed_reps)]), 0)
        ens = M.m_ensemble(ctx["view"], ridge_alpha=100.0, knn_k=10, pls_n=10, n_comp=128)
        yp = w_embed * emb + (1.0 - w_embed) * ens
        yp = np.clip(yp, ctx["lo"], ctx["hi"]).astype(np.float32)
        parts.append((ctx["ids"], yp))
        print(f"  {target:14s} embed mean|.|={np.abs(emb).mean():.3f} "
              f"ens mean|.|={np.abs(ens).mean():.3f} blend mean|.|={np.abs(yp).mean():.3f}",
              flush=True)

    print(f"first_round_blend: recipe={args.recipe} w_embed={w_embed} "
          f"reps={args.embed_reps}")
    assemble_submission(assets, parts, args.out)


if __name__ == "__main__":
    main()
