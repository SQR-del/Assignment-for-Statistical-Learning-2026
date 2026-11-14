"""Donor-split labels, split-half technical noise, and label-noise floors."""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

from _common import (CELLS, DMSO, OUT, PRIOR, ensure_out, load_de,
                     load_genes, logcpm, paired_effects, signed_from_effect)

ensure_out()
GENES = load_genes()
NG = len(GENES)
DE = load_de()
FULL = pd.read_parquet(os.path.join(OUT, "pseudobulk_long.parquet"))
HALF = pd.read_parquet(os.path.join(OUT, "halfpseudobulk_long.parquet"))


def pivot(df, cell, half=False):
    sub = df[df.cell_type == cell]
    idx = ["donor_id", "half", "sm_name"] if half else ["donor_id", "sm_name"]
    m = sub.pivot_table(index=idx, columns="gene", values="counts",
                        aggfunc="sum", fill_value=0)
    return logcpm(m.reindex(columns=GENES, fill_value=0))





rows = []
calib = []
halffull = []
repro_rows = []
for cell in CELLS:
    L = pivot(FULL, cell)
    H = pivot(HALF, cell, half=True)
    donors = sorted(FULL.loc[FULL.cell_type == cell, "donor_id"].unique())
    drugs = [d for d in sorted(DE.loc[DE.cell_type == cell, "sm_name"].unique())]
    D = len(drugs)
    de_mat = DE[DE.cell_type == cell].set_index("sm_name").loc[drugs, GENES].to_numpy(float)
    donor_lab = np.full((len(donors), D, NG), np.nan, np.float32)
    half_lab = np.full((len(donors), 2, D, NG), np.nan, np.float32)
    full_lab = np.full((D, NG), np.nan, np.float32)
    for i, drug in enumerate(drugs):
        eff, dons = paired_effects(L, drug)
        if len(dons) < 2:
            continue
        mean = np.nanmean(eff, 0)
        var = np.nanvar(eff, 0, ddof=1)
        se = np.sqrt(np.maximum(var / len(dons), 1e-12))
        full_lab[i] = signed_from_effect(mean, se, len(dons) - 1)
        gm = np.isfinite(de_mat[i]) & np.isfinite(full_lab[i])
        if gm.sum() > 10:
            rr = np.corrcoef(de_mat[i][gm], full_lab[i][gm])[0, 1]
            repro_rows.append({
                "cell_type": cell, "drug": drug,
                "pearson": float(rr),
                "spearman": float(stats.spearmanr(de_mat[i][gm], full_lab[i][gm]).statistic),
                "slope_de_on_repro": float(np.polyfit(full_lab[i][gm], de_mat[i][gm], 1)[0]),
            })
        for j, d in enumerate(donors):
            if d not in dons:
                continue
            donor_lab[j, i] = signed_from_effect(eff[dons.index(d)], se, len(dons) - 1)
            for h in [0, 1]:
                try:
                    e = (H.loc[(d, h, drug)] - H.loc[(d, h, DMSO)]).to_numpy(float)
                except KeyError:
                    continue
                half_lab[j, h, i] = signed_from_effect(e, se, len(dons) - 1)
    ok = np.isfinite(full_lab).all(1) & np.isfinite(de_mat).all(1)
    a, b = np.polyfit(full_lab[ok].ravel(), de_mat[ok].ravel(), 1)
    r = np.corrcoef(full_lab[ok].ravel(), de_mat[ok].ravel())[0, 1]
    rho = stats.spearmanr(full_lab[ok].ravel(), de_mat[ok].ravel()).statistic
    calib.append({"cell_type": cell, "slope_de_on_repro": a, "intercept": b,
                  "pearson": r, "spearman": rho, "n_drug": int(ok.sum())})
    n_don = np.isfinite(donor_lab).sum(0).astype(float)
    donor_var = np.nanvar(donor_lab, axis=0, ddof=1)
    donor_noise = np.where(n_don >= 2, donor_var / np.maximum(n_don, 1), np.nan) * (a ** 2)
    hdiff = (half_lab[:, 0] - half_lab[:, 1]) ** 2
    tech_full = np.nanmean(hdiff / 4.0, axis=0)
    tech_noise = tech_full / 3.0 * (a ** 2)
    small = half_lab[:, 0]
    small_mean = np.nanmean(small, axis=0)
    np.savez_compressed(os.path.join(OUT, "repro_" + cell.replace(" ", "_").replace("+", "p") + ".npz"),
                        drugs=np.array(drugs), full=full_lab.astype(np.float32),
                        small=small_mean.astype(np.float32))
    for i, drug in enumerate(drugs):
        good = np.isfinite(small_mean[i]) & np.isfinite(full_lab[i])
        if good.sum() > 10:
            aa = np.polyfit(small_mean[i][good], full_lab[i][good], 1)[0]
            rr = np.corrcoef(small_mean[i][good], full_lab[i][good])[0, 1]
            amp = np.abs(full_lab[i][good]).mean() / max(np.abs(small_mean[i][good]).mean(), 1e-9)
        else:
            aa = rr = amp = np.nan
        halffull.append({"cell_type": cell, "drug": drug, "slope_large_on_small": aa,
                         "corr_small_large": rr, "amp_ratio": amp,
                         "mse_small_vs_full": float(np.nanmean((small_mean[i] - full_lab[i]) ** 2)),
                         "mean_abs_diff": float(np.nanmean(np.abs(small_mean[i] - full_lab[i]))),
                         "mean_abs_small": float(np.nanmean(np.abs(small_mean[i]))),
                         "mean_abs_full": float(np.nanmean(np.abs(full_lab[i])))})
        rows.append({"cell_type": cell, "drug": drug,
                     "noise_donor": float(np.nanmean(donor_noise[i])),
                     "noise_technical": float(np.nanmean(tech_noise[i])),
                     "label_mse_vs_de": float(np.nanmean((de_mat[i] - full_lab[i]) ** 2)),
                     "mean_abs_de": float(np.nanmean(np.abs(de_mat[i]))),
                     "n_genes": int(np.isfinite(donor_noise[i]).sum())})
    print(f"{cell}: repro r={r:.3f} rho={rho:.3f} slope={a:.3f} "
          f"noise donor={np.nanmean(donor_noise):.3f} tech={np.nanmean(tech_noise):.3f}")

row_df = pd.DataFrame(rows)
row_df.to_csv(os.path.join(OUT, "noise_by_row.csv"), index=False)
pd.DataFrame(calib).to_csv(os.path.join(OUT, "repro_calibration.csv"), index=False)
hf = pd.DataFrame(halffull)
hf.to_csv(os.path.join(OUT, "half_vs_full.csv"), index=False)
repro = pd.DataFrame(repro_rows)
repro.to_csv(os.path.join(OUT, "repro_by_row.csv"), index=False)
print("repro per-row")
print(repro.groupby("cell_type")[["pearson", "spearman", "slope_de_on_repro"]].mean().round(3).to_string())
print("repro calibration")
print(pd.DataFrame(calib).to_string(index=False))
print("half vs full")
print(hf[["slope_large_on_small", "corr_small_large", "amp_ratio"]].describe().round(3).to_string())
summary = {
    "prior": PRIOR,
    "mean_noise_donor": float(row_df.noise_donor.mean()),
    "mean_noise_technical": float(row_df.noise_technical.mean()),
    "mean_abs_de": float(row_df.mean_abs_de.mean()),
    "mrrmse_floor_donor": float(np.mean(np.sqrt(row_df.noise_donor))),
    "mrrmse_floor_technical": float(np.mean(np.sqrt(row_df.noise_technical))),
    "rms_label": float(np.sqrt((row_df.mean_abs_de ** 2).mean())),
    "half_vs_full": hf[["slope_large_on_small", "corr_small_large", "amp_ratio"]].mean().to_dict(),
    "calibration": pd.DataFrame(calib).to_dict(orient="records"),
}
with open(os.path.join(OUT, "noise_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in summary.items() if k != "calibration"}, indent=2))
print("saved", OUT)
