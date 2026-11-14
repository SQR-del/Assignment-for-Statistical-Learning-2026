"""Cell-level Limma-style DE (large-sample model) from sufficient statistics."""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

from _common import CELLS, DMSO, OUT, ensure_out, load_de, load_genes

ensure_out()
GENES = load_genes()
NG = len(GENES)
KEYS = pd.read_csv(os.path.join(OUT, "group_keys.csv"))
STATS = pd.read_parquet(os.path.join(OUT, "cell_stats.parquet"))
DE = load_de()


def build_pivots(cell):
    gids = KEYS.loc[KEYS.cell_type == cell, "group_id"].to_numpy()
    sub = STATS[STATS.group_id.isin(gids)].merge(KEYS, on="group_id", how="left")
    sub["mean"] = sub.sum_x / sub.n
    sub["var"] = (sub.sum_x2 - sub.n * sub["mean"] ** 2) / (sub.n - 1).clip(lower=1)
    sub["var"] = sub["var"].clip(lower=0)
    sub["key"] = sub.donor_id + "|" + sub.sm_name
    out = {}
    for col in ["mean", "var", "n"]:
        piv = sub.set_index(["key", "gene_id"])[col].unstack("gene_id")
        piv = piv.reindex(columns=np.arange(1, NG + 1), fill_value=0)
        out[col] = piv
    return out


def combine_donors(piv, drug, donors):
    keys = [f"{d}|{drug}" for d in donors]
    kd = [f"{d}|{DMSO}" for d in donors]
    md = piv["mean"].reindex(keys).to_numpy(float)
    vd = piv["var"].reindex(keys).to_numpy(float)
    nd = piv["n"].reindex(keys).to_numpy(float)
    mc = piv["mean"].reindex(kd).to_numpy(float)
    vc = piv["var"].reindex(kd).to_numpy(float)
    nc = piv["n"].reindex(kd).to_numpy(float)
    valid = (nd >= 2) & (nc >= 2)
    diff = md - mc
    vdiff = np.where(valid, vd / np.maximum(nd, 1) + vc / np.maximum(nc, 1), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / vdiff
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    sw = w.sum(0)
    effect = np.where(sw > 0, (w * np.nan_to_num(diff)).sum(0) / sw, 0.0)
    se = np.where(sw > 0, np.sqrt(1.0 / sw), np.inf)
    dfd = np.where(valid,
                   (vd / np.maximum(nd, 1) + vc / np.maximum(nc, 1)) ** 2 /
                   ((vd / np.maximum(nd, 1)) ** 2 / np.maximum(nd - 1, 1) +
                    (vc / np.maximum(nc, 1)) ** 2 / np.maximum(nc - 1, 1)),
                   1.0)
    dfc = np.where(sw > 0, (sw ** 2) / np.maximum((w ** 2 / np.maximum(dfd, 1)).sum(0), 1e-12), 1.0)
    dfc = np.clip(dfc, 1.0, 1e6)
    t = np.where(se > 0, effect / se, 0.0)
    p = 2 * stats.t.sf(np.abs(t), df=dfc)
    p = np.where((se > 0) & np.isfinite(p), p, 1.0)
    signed = -np.log10(np.maximum(p, 1e-300)) * np.sign(effect)
    return signed, effect, p, dfc


def per_donor(piv, drug, donors):
    keys = [f"{d}|{drug}" for d in donors]
    kd = [f"{d}|{DMSO}" for d in donors]
    md = piv["mean"].reindex(keys).to_numpy(float)
    vd = piv["var"].reindex(keys).to_numpy(float)
    nd = piv["n"].reindex(keys).to_numpy(float)
    mc = piv["mean"].reindex(kd).to_numpy(float)
    vc = piv["var"].reindex(kd).to_numpy(float)
    nc = piv["n"].reindex(kd).to_numpy(float)
    valid = (nd >= 2) & (nc >= 2)
    diff = md - mc
    var = np.where(valid, vd / np.maximum(nd, 1) + vc / np.maximum(nc, 1), np.nan)
    se = np.sqrt(np.where(valid, var, np.inf))
    t = np.where(se > 0, diff / se, 0.0)
    dfd = np.where(valid,
                   var ** 2 / ((vd / np.maximum(nd, 1)) ** 2 / np.maximum(nd - 1, 1) +
                               (vc / np.maximum(nc, 1)) ** 2 / np.maximum(nc - 1, 1)),
                   1.0)
    dfd = np.clip(dfd, 1.0, 1e6)
    p = 2 * stats.t.sf(np.abs(t), df=dfd)
    p = np.where(valid, p, 1.0)
    signed = -np.log10(np.maximum(p, 1e-300)) * np.sign(diff)
    return np.where(valid, signed, np.nan), valid


def evaluate(cell, signed, drugs):
    rows = []
    for i, d in enumerate(drugs):
        hit = DE[(DE.cell_type == cell) & (DE.sm_name == d)]
        if len(hit) == 0:
            continue
        y = hit.iloc[0][GENES].to_numpy(float)
        x = signed[i]
        rows.append({"cell_type": cell, "drug": d,
                     "pearson": float(np.corrcoef(y, x)[0, 1]),
                     "spearman": float(stats.spearmanr(y, x).statistic),
                     "slope_de_on_repro": float(np.polyfit(x, y, 1)[0]),
                     "mean_abs_de": float(np.abs(y).mean()),
                     "mean_abs_repro": float(np.abs(x).mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    cells = [only] if only else CELLS
    all_eval = []
    all_labels = {}
    all_donor = {}
    drugs = sorted(DE.sm_name.unique())
    for cell in cells:
        piv = build_pivots(cell)
        donors = sorted(set(KEYS[KEYS.cell_type == cell].donor_id))
        n_drugs = len(drugs)
        signed = np.zeros((n_drugs, NG), np.float32)
        donor_labels = np.zeros((len(donors), n_drugs, NG), np.float32)
        for i, d in enumerate(drugs):
            s, eff, p, dfc = combine_donors(piv, d, donors)
            signed[i] = s
            dl, valid = per_donor(piv, d, donors)
            donor_labels[:, i] = dl
        all_labels[cell] = signed
        all_donor[cell] = donor_labels
        ev = evaluate(cell, signed, drugs)
        all_eval.append(ev)
        print(f"{cell}: mean row pearson={ev.pearson.mean():.3f} "
              f"spearman={ev.spearman.mean():.3f} "
              f"slope={ev.slope_de_on_repro.median():.3f} "
              f"n_drugs={len(ev)}")
    ev = pd.concat(all_eval, ignore_index=True)
    ev.to_csv(os.path.join(OUT, "cell_limma_eval.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "cell_limma_labels.npz"),
                        cells=np.array(cells), drugs=np.array(drugs),
                        **{c: all_labels[c] for c in cells})
    for c in cells:
        np.savez_compressed(os.path.join(OUT, f"donor_labels_{c.replace(' ','_').replace('+','p')}.npz"),
                            donors=np.array(sorted(set(KEYS[KEYS.cell_type == c].donor_id))),
                            drugs=np.array(drugs), labels=all_donor[c])
    print(ev[["pearson", "spearman", "slope_de_on_repro"]].describe().to_string())
