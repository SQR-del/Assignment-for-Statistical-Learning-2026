"""Model blending / averaging utilities.

`learn_simplex_weights` fits non-negative weights that sum to 1 by minimising the
official metric (mean row-wise RMSE) on a validation set — used to combine
decorrelated model families (see REPORT.md §4.4).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import common as C


def blend(P, weights):
    """Weighted sum of stacked predictions.  P: (n, C, G) or list of (n, G)."""
    P = np.asarray(P)
    w = np.asarray(weights, dtype=np.float64)
    if P.ndim == 3:
        return np.einsum("nkg,k->ng", P, w)
    raise ValueError("P must be (n, C, G)")


def learn_simplex_weights(P, y, maxiter=2000, xatol=1e-4, fatol=1e-6, seed=0):
    """Fit weights w (sum to 1, non-negative) minimising mrrmse of P @ w on y.

    P: (n, C, G) component predictions; y: (n, G).  Uses a softmax reparametrisation
    so the simplex constraint is automatic.
    """
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    C_ = P.shape[1]

    def obj(a):
        w = np.exp(a); w = w / w.sum()
        return C.mrrmse(np.einsum("nkg,k->ng", P, w), y)

    res = minimize(obj, np.zeros(C_), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": xatol, "fatol": fatol})
    w = np.exp(res.x); w = w / w.sum()
    return w, float(res.fun)


def fixed_blend(preds, weights):
    """Fixed-weight average of a list of (n, G) predictions."""
    return sum(w * np.asarray(p, np.float32) for w, p in zip(weights, preds)).astype(np.float32)
