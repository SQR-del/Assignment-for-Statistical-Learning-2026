"""Method-selection experiments for reproducing the Limma labels.

Merges the former prototype scripts into one entry point:

    prototype : donor-level vs plate-level pseudobulk (CD4+)
    sweep     : normalization / prior-count sweep (CD4+)
    hybrid    : pseudobulk effect + cell-level standard error (CD4+)

Outputs (in ``outputs/``):
    prototype_TCD4.csv
    prototype_plate_TCD4_prior0.5.csv, ..._prior1.0.csv
    normalization_sweep.csv
    hybrid_TCD4.csv

Run:  python 04_method_comparison.py [--analysis all|prototype|sweep|hybrid]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from _common import DMSO, OUT, logcpm, load_de, load_genes

CELL = "T cells CD4+"
GENES = load_genes()
NG = len(GENES)
DE = load_de()


def pivot_counts(df, cell, level):
    idx = ["donor_id", "sm_name"] if level == "donor" else [
        "plate_name", "sm_name"]
    sub = df[df.cell_type == cell]
    m = sub.pivot_table(index=idx, columns="gene", values="counts",
                        aggfunc="sum", fill_value=0)
    return m.reindex(columns=GENES, fill_value=0)


def signed_rows(L):
    """One signed -log10(p) vector per drug, paired against DMSO."""
    dmso = L.xs(DMSO, level="sm_name")
    rows = []
    for drug in sorted(set(L.index.get_level_values(1)) - {DMSO}):
        cur = L.xs(drug, level="sm_name")
        units = [u for u in cur.index if u in dmso.index]
        if len(units) < 2:
            continue
        diff = cur.loc[units].to_numpy(float) - dmso.loc[units].to_numpy(float)
        n = len(units)
        mean = diff.mean(0)
        var = diff.var(0, ddof=1)
        t = mean / np.sqrt(np.maximum(var / n, 1e-12))
        p = 2 * stats.t.sf(np.abs(t), df=n - 1)
        p = np.where(var <= 1e-12, 1.0, p)
        rows.append((drug, n, -np.log10(np.maximum(p, 1e-300)) * np.sign(mean)))
    return rows


def evaluate(cell, rows, ncol="n_donors", scol="slope_de_on_repro"):
    out = []
    for drug, n_units, signed in rows:
        hit = DE[(DE.cell_type == cell) & (DE.sm_name == drug)]
        if len(hit) == 0:
            continue
        y = hit.iloc[0][GENES].to_numpy(float)
        out.append({
            "drug": drug, ncol: n_units,
            "pearson": float(np.corrcoef(y, signed)[0, 1]),
            "spearman": float(stats.spearmanr(y, signed).statistic),
            scol: float(np.polyfit(signed, y, 1)[0]),
            "mean_abs_de": float(np.abs(y).mean()),
            "mean_abs_repro": float(np.abs(signed).mean()),
        })
    return pd.DataFrame(out)


def run_prototype():
    donor = pd.read_parquet(os.path.join(OUT, "pseudobulk_long.parquet"))
    mat = logcpm(pivot_counts(donor, CELL, "donor"), prior=0.5)
    res = evaluate(CELL, signed_rows(mat))
    res.to_csv(os.path.join(OUT, "prototype_TCD4.csv"), index=False)
    print("[donor] " + res[["pearson", "spearman",
                            "slope_de_on_repro"]].describe().round(3).to_string())

    plate = pd.read_parquet(os.path.join(OUT, "pseudobulk_plate_long.parquet"))
    mat = pivot_counts(plate, CELL, "plate")
    for prior in [0.5, 1.0]:
        res = evaluate(CELL, signed_rows(logcpm(mat, prior=prior)),
                       ncol="n_plates", scol="slope")
        res.to_csv(os.path.join(OUT, f"prototype_plate_TCD4_prior{prior}.csv"),
                   index=False)
        print(f"[plate prior={prior}] pearson={res.pearson.mean():.3f} "
              f"spearman={res.spearman.mean():.3f} "
              f"slope={res.slope.median():.3f}")


def run_sweep():
    raw = pd.read_parquet(os.path.join(OUT, "pseudobulk_long.parquet"))
    norm = pd.read_parquet(os.path.join(OUT, "normpseudobulk_long.parquet"))
    counts = pd.read_csv(os.path.join(OUT, "obs_counts.csv"))
    rawm = pivot_counts(raw, CELL, "donor")
    normm = pd.DataFrame(
        norm[norm.cell_type == CELL].pivot_table(
            index=["donor_id", "sm_name"], columns="gene", values="norm_sum",
            aggfunc="sum", fill_value=0).reindex(columns=GENES, fill_value=0))
    cnt = counts[counts.cell_type == CELL].set_index(
        ["donor_id", "sm_name"]).n_cells
    cnt = cnt.reindex(rawm.index).fillna(1).to_numpy(float)[:, None]

    variants = []
    for prior in [0.1, 0.5, 1.0, 5.0]:
        lib = rawm.sum(axis=1).to_numpy(float)
        variants.append((f"raw CPM prior={prior}", pd.DataFrame(
            np.log2((rawm.to_numpy(float) + prior) / lib[:, None] * 1e6),
            index=rawm.index, columns=GENES)))
    variants.append(("raw log2(x+1)", pd.DataFrame(
        np.log2(rawm.to_numpy(float) + 1), index=rawm.index, columns=GENES)))
    variants.append(("raw mean count log2(x+1)", pd.DataFrame(
        np.log2(rawm.to_numpy(float) / cnt + 1),
        index=rawm.index, columns=GENES)))
    variants.append(("norm_sum log2(x+1)", pd.DataFrame(
        np.log2(normm.to_numpy(float) + 1), index=normm.index, columns=GENES)))
    libn = normm.sum(axis=1).to_numpy(float)
    variants.append(("norm_sum logCPM", pd.DataFrame(
        np.log2((normm.to_numpy(float) + 0.5) / libn[:, None] * 1e6),
        index=normm.index, columns=GENES)))
    variants.append(("norm mean log2(x+1)", pd.DataFrame(
        np.log2(normm.to_numpy(float) / cnt + 1),
        index=normm.index, columns=GENES)))

    rows = []
    for label, mat in variants:
        res = evaluate(CELL, signed_rows(mat))
        rows.append({"normalization": label,
                     "pearson": res.pearson.mean(),
                     "spearman": res.spearman.mean(),
                     "slope_median": float(res.slope_de_on_repro.median())})
        print(f"{label:32s} pearson={rows[-1]['pearson']:.3f} "
              f"spearman={rows[-1]['spearman']:.3f} "
              f"slope_med={rows[-1]['slope_median']:.3f}")
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT, "normalization_sweep.csv"), index=False)


def run_hybrid():
    genes = GENES
    keys = pd.read_csv(os.path.join(OUT, "group_keys.csv"))
    stats_df = pd.read_parquet(os.path.join(OUT, "cell_stats.parquet"))
    pb = pd.read_parquet(os.path.join(OUT, "pseudobulk_long.parquet"))

    gmap = pd.DataFrame({"gene": genes, "gene_id": np.arange(1, NG + 1)})
    gids = keys.loc[keys.cell_type == CELL, "group_id"].to_numpy()
    st = stats_df[stats_df.group_id.isin(gids)].merge(
        keys, on="group_id", how="left").merge(gmap, on="gene_id", how="left")
    st["mean"] = st.sum_x / st.n
    st["var"] = ((st.sum_x2 - st.n * st["mean"] ** 2) /
                 (st.n - 1).clip(lower=1)).clip(lower=0)

    mat = pivot_counts(pb, CELL, "donor")
    L = logcpm(mat, prior=0.1)
    dmso = L.xs(DMSO, level="sm_name")
    piv = {}
    for col in ["mean", "var", "n"]:
        p = st.set_index(["donor_id", "sm_name", "gene"]).loc[:, col].unstack("gene")
        piv[col] = p.reindex(columns=genes, fill_value=0)

    donors = sorted(st.donor_id.unique())
    drugs = sorted(DE.sm_name.unique())
    rows = []
    for drug in drugs:
        labels = []
        for d in donors:
            try:
                eff = (L.loc[(d, drug)] - dmso.loc[d]).to_numpy(float)
            except KeyError:
                continue
            md = piv["mean"].loc[(d, drug)].to_numpy(float)
            vd = piv["var"].loc[(d, drug)].to_numpy(float)
            nd = piv["n"].loc[(d, drug)].to_numpy(float)
            vc = piv["var"].loc[(d, DMSO)].to_numpy(float)
            nc = piv["n"].loc[(d, DMSO)].to_numpy(float)
            valid = (nd >= 2) & (nc >= 2)
            se = np.sqrt(np.where(
                valid, vd / np.maximum(nd, 1) + vc / np.maximum(nc, 1), np.inf))
            t = np.where(se > 0, eff / se, 0.0)
            p = 2 * stats.t.sf(np.abs(t), df=np.maximum(nd + nc - 2, 1))
            p = np.where(valid, p, 1.0)
            labels.append(np.where(
                valid, -np.log10(np.maximum(p, 1e-300)) * np.sign(eff), np.nan))
        if len(labels) < 2:
            continue
        Lm = np.array(labels)
        if not np.isfinite(Lm).any():
            continue
        meanl = np.nanmean(Lm, axis=0)
        if np.nanstd(meanl) == 0 or not np.isfinite(meanl).any():
            continue
        meanl = np.nan_to_num(meanl, nan=0.0, posinf=0.0, neginf=0.0)
        hit = DE[(DE.cell_type == CELL) & (DE.sm_name == drug)]
        if len(hit) == 0:
            continue
        y = hit.iloc[0][genes].to_numpy(float)
        rows.append({"drug": drug,
                     "pearson": np.corrcoef(y, meanl)[0, 1],
                     "spearman": stats.spearmanr(y, meanl).statistic,
                     "slope": np.polyfit(meanl, y, 1)[0],
                     "mean_abs_de": np.abs(y).mean(),
                     "mean_abs_hybrid": np.abs(meanl).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "hybrid_TCD4.csv"), index=False)
    print("[hybrid] " + res[["pearson", "spearman", "slope"]]
          .describe().round(3).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="all",
                        choices=["all", "prototype", "sweep", "hybrid"])
    args = parser.parse_args()
    if args.analysis in ("all", "prototype"):
        run_prototype()
    if args.analysis in ("all", "sweep"):
        run_sweep()
    if args.analysis in ("all", "hybrid"):
        run_hybrid()


if __name__ == "__main__":
    main()
