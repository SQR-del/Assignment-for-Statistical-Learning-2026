"""Decompose model error into direction, magnitude and label noise."""
import os
import glob
import numpy as np
import pandas as pd

from _common import CELLS, OUT, ROOT, ensure_out

ensure_out()
CELLS = CELLS[:4]
NOISE = pd.read_csv(os.path.join(OUT, "noise_by_row.csv"))


def load_preds():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "predictions", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        rows = z["rows"]
        cell = str(rows[0, 0])
        if cell not in CELLS:
            continue
        out[cell] = {"drugs": [str(r[1]) for r in rows],
                     "y": z["y_test"].astype(np.float64),
                     "models": {k: z[k].astype(np.float64) for k in z.files
                                if k not in ["y_test", "genes", "rows"]}}
    return out


def parts(y, pred):
    total = np.mean((y - pred) ** 2, axis=1)
    denom = np.sum(pred ** 2, axis=1)
    a = np.where(denom > 0, np.sum(y * pred, axis=1) / denom, 0.0)
    direction = np.mean((y - a[:, None] * pred) ** 2, axis=1)
    magnitude = np.mean(((a - 1)[:, None] * pred) ** 2, axis=1)
    return total, direction, magnitude, a


preds = load_preds()
rows = []
for cell, d in preds.items():
    nmap = NOISE[NOISE.cell_type == cell].set_index("drug")
    noise = {}
    for kind in ["technical", "donor"]:
        col = f"noise_{kind}"
        v = np.array([nmap.loc[x, col] if x in nmap.index else np.nan
                      for x in d["drugs"]], float)
        v = np.where(np.isfinite(v), v, np.nanmedian(v))
        noise[kind] = v
    for model, pred in d["models"].items():
        total, direction, magnitude, a = parts(d["y"], pred)
        for kind in ["technical", "donor"]:
            n = noise[kind]
            mag_mse = float(magnitude.mean())
            dir_mse = float(direction.mean())
            noise_mse = float(n.mean())
            tot_mse = float(total.mean())
            noise_used = min(noise_mse, dir_mse)
            rows.append({
                "cell_type": cell, "model": model, "noise_kind": kind,
                "mrrmse": float(np.mean(np.sqrt(total))),
                "mse": tot_mse,
                "magnitude_share": mag_mse / tot_mse,
                "noise_share": noise_used / tot_mse,
                "direction_share": (dir_mse - noise_used) / tot_mse,
                "direction_raw_share": dir_mse / tot_mse,
                "noise_over_total": noise_mse / tot_mse,
                "magnitude_minus_direction": (mag_mse - dir_mse) / tot_mse,
                "alpha_median_row": float(np.median(a)),
                "noise_rmse": float(np.mean(np.sqrt(n))),
                "magnitude_rmse": float(np.mean(np.sqrt(magnitude))),
                "direction_rmse": float(np.mean(np.sqrt(direction))),
            })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "error_decomposition.csv"), index=False)
for kind in ["technical", "donor"]:
    sub = df[df.noise_kind == kind]
    print(f"=== {kind} noise ===")
    print(sub.groupby("model")[["mrrmse", "magnitude_share", "noise_share",
                                "direction_share", "direction_raw_share",
                                "noise_over_total", "alpha_median_row"]]
          .mean().round(4).to_string())
    print()
    print(sub.groupby("cell_type")[["mrrmse", "magnitude_share", "noise_share",
                                    "direction_share", "direction_raw_share"]]
          .mean().round(4).to_string())
    print()
