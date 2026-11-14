"""Shared constants and helpers for the Limma reproduction / noise analysis.

All scripts in this folder write to ``<data_analysis>/limma/outputs/`` and read
the raw competition files from ``<project_root>/data/``.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.dirname(HERE)
ROOT = os.path.dirname(PACKAGE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(OUT, "figures")

CELLS = ["T cells CD4+", "T regulatory cells", "T cells CD8+", "NK cells",
         "B cells", "Myeloid cells"]
FULL = CELLS[:4]
SHORT = {"T cells CD4+": "CD4+", "T regulatory cells": "Treg",
         "T cells CD8+": "CD8+", "NK cells": "NK",
         "B cells": "B", "Myeloid cells": "Myeloid"}
DMSO = "Dimethyl Sulfoxide"
PRIOR = 0.1
SEED = 20251001


def ensure_out():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)


def load_genes():
    return pd.read_csv(os.path.join(OUT, "genes.csv")).gene.tolist()


def load_de(drop_controls=True):
    df = pd.read_parquet(os.path.join(DATA, "de_train.parquet"))
    if drop_controls:
        df = df[~df.control].reset_index(drop=True)
    return df


def logcpm(mat, prior=PRIOR):
    """log2 CPM of a (unit x gene) count matrix with a fixed prior count."""
    lib = mat.sum(axis=1).to_numpy(float)
    return pd.DataFrame(
        np.log2((mat.to_numpy(float) + prior) / lib[:, None] * 1e6),
        index=mat.index, columns=mat.columns)


def signed_from_effect(eff, se, df):
    """Signed -log10(p) from an effect size and a standard error."""
    t = np.where(se > 0, eff / se, 0.0)
    p = 2 * stats.t.sf(np.abs(t), df=max(df, 1))
    p = np.where(np.isfinite(p), p, 1.0)
    return -np.log10(np.maximum(p, 1e-300)) * np.sign(eff)


def paired_effects(L, drug, units=DMSO):
    """Drug-minus-DMSO effect per unit (donor or plate) for one drug."""
    dmso = L.xs(DMSO, level="sm_name")
    cur = L.xs(drug, level="sm_name")
    keep = [u for u in cur.index if u in dmso.index]
    return cur.loc[keep].to_numpy(float) - dmso.loc[keep].to_numpy(float), keep


def mrrmse(pred, y):
    pred = np.asarray(pred, np.float64)
    y = np.asarray(y, np.float64)
    return float(np.mean(np.sqrt(np.mean((y - pred) ** 2, axis=1))))
