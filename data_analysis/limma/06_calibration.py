"""Test whether amplitude calibration improves P1 predictions."""
import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from _common import CELLS, OUT, ROOT, ensure_out, mrrmse as mrr

ensure_out()
CELLS = CELLS[:4]


data = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results", "predictions", "*.npz"))):
    z = np.load(f, allow_pickle=True)
    rows = z["rows"]
    cell = str(rows[0, 0])
    if cell not in CELLS:
        continue
    data[cell] = {k: z[k] for k in z.files
                  if k not in ["y_test", "genes", "rows"]}
    data[cell]["__y"] = z["y_test"]

models = sorted(set.intersection(*[set(data[c]) for c in CELLS]) - {"__y"})
hf = pd.read_csv(os.path.join(OUT, "half_vs_full.csv"))
cell_slope = hf.groupby("cell_type").slope_large_on_small.mean().to_dict()
cell_amp = hf.groupby("cell_type").amp_ratio.mean().to_dict()
print("models", models)
rows = []
for model in models:
    for cell in CELLS:
        pred = np.asarray(data[cell][model], np.float64)
        y = np.asarray(data[cell]["__y"], np.float64)
        base = mrr(pred, y)
        other = [c for c in CELLS if c != cell]
        ptr = np.concatenate([np.asarray(data[c][model], np.float64) for c in other])
        ytr = np.concatenate([np.asarray(data[c]["__y"], np.float64) for c in other])

        def f(alpha):
            return mrr(alpha * ptr, ytr)

        best = minimize_scalar(f, bounds=(0.05, 1.6), method="bounded")
        a_cv = float(best.x)
        a_oracle = float(minimize_scalar(lambda a: mrr(a * pred, y),
                                         bounds=(0.05, 1.6), method="bounded").x)
        rows.append({
            "model": model, "cell_type": cell, "base_mrrmse": base,
            "alpha_cv": a_cv, "mrrmse_cv": mrr(a_cv * pred, y),
            "alpha_oracle": a_oracle, "mrrmse_oracle": mrr(a_oracle * pred, y),
            "alpha_fixed_slope": 0.808, "mrrmse_slope": mrr(0.808 * pred, y),
            "alpha_fixed_inv": 1.238, "mrrmse_inv": mrr(1.238 * pred, y),
            "alpha_fixed_amp": 0.955, "mrrmse_amp": mrr(0.955 * pred, y),
            "alpha_cell_slope": cell_slope[cell],
            "mrrmse_cell_slope": mrr(cell_slope[cell] * pred, y),
            "alpha_cell_amp": cell_amp[cell],
            "mrrmse_cell_amp": mrr(cell_amp[cell] * pred, y),
        })
df = pd.DataFrame(rows)
df["delta_cv"] = df.mrrmse_cv - df.base_mrrmse
df["delta_oracle"] = df.mrrmse_oracle - df.base_mrrmse
df.to_csv(os.path.join(OUT, "calibration_results.csv"), index=False)
print(df.groupby("model")[["base_mrrmse", "mrrmse_cv", "mrrmse_oracle",
                           "alpha_cv", "alpha_oracle", "delta_cv",
                           "mrrmse_slope", "mrrmse_inv", "mrrmse_amp",
                           "mrrmse_cell_slope", "mrrmse_cell_amp"]]
      .mean().round(4).to_string())

rng = np.random.default_rng(20251001)
boot = []
for model in models:
    for cell in CELLS:
        pred = np.asarray(data[cell][model], np.float64)
        y = np.asarray(data[cell]["__y"], np.float64)
        a = float(df[(df.model == model) & (df.cell_type == cell)].alpha_cv.iloc[0])
        d = np.sqrt(np.mean((y - a * pred) ** 2, axis=1)) - \
            np.sqrt(np.mean((y - pred) ** 2, axis=1))
        idx = rng.integers(0, len(d), size=(1000, len(d)))
        bs = d[idx].mean(1)
        boot.append({"model": model, "cell_type": cell,
                     "delta_mean": float(d.mean()),
                     "ci_lo": float(np.quantile(bs, 0.025)),
                     "ci_hi": float(np.quantile(bs, 0.975))})
boot = pd.DataFrame(boot)
boot.to_csv(os.path.join(OUT, "calibration_bootstrap.csv"), index=False)
print("bootstrap delta (cv)")
print(boot.groupby("model")[["delta_mean", "ci_lo", "ci_hi"]].mean().round(4).to_string())
