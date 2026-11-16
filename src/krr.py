"""Null-centred anchor-analog kernel ridge regression (KRR), the "own framework" component of the blend.

This is the local KRR (one model per target cell type, labels = its 15 anchors); the joint KRR is krr_joint.py.

For target cell type t, anchors A (compounds observed in t) and query compound d:
    yhat_d = r_t + c * xbar_d + k(d, A) (K_AA + alpha * tr(K)/n * I)^{-1} [Y_tA - r_t - c * xbar_A]
r_t    : per-gene median of the target's anchors (null reference)
x_s,d  : source-s response minus that source's per-gene median over all shared compounds
k(., .): linear kernel on the concatenated source profiles; xbar = mean over sources
v1     : c=0.2, alpha=10 (selected on protocol-A inner tasks)
v2     : c=0.4, alpha=3, xbar projected on the top-64 source components, output clipped to the visible range
The positive controls (Belinostat, Dabrafenib) are excluded everywhere (`skip`), as in the submitted KRR models.
"""
import numpy as np

CONFIGS = {1: dict(c=0.2, alpha=10.0, K=None, clip=False), 2: dict(c=0.4, alpha=3.0, K=64, clip=True)}


def krr_predict(Y, row_of, n_comps, t, src, anchors, qcomps, version=2, skip=()):
    cfg = CONFIGS[version]; c, alpha = cfg['c'], cfg['alpha']
    anchors = [d for d in anchors if d not in skip and all((s, d) in row_of for s in src) and (t, d) in row_of]
    common = [d for d in range(n_comps) if d not in skip and all((s, d) in row_of for s in src)]
    rS = np.stack([np.median(Y[[row_of[(s, d)] for d in common]], 0) for s in src])
    YA = Y[[row_of[(t, d)] for d in anchors]]; rt = np.median(YA, 0)
    XA = np.stack([Y[[row_of[(s, d)] for d in anchors]] - rS[j] for j, s in enumerate(src)])
    XQ = np.stack([Y[[row_of[(s, d)] for d in qcomps]] - rS[j] for j, s in enumerate(src)])
    n = len(anchors)
    FA = XA.transpose(1, 0, 2).reshape(n, -1); FQ = XQ.transpose(1, 0, 2).reshape(len(qcomps), -1); K = FA @ FA.T
    if cfg['K'] is None:
        xA, xQ = XA.mean(0), XQ.mean(0)
    else:
        Cm = np.concatenate([Y[[row_of[(s, d)] for d in common]] - rS[j] for j, s in enumerate(src)])
        ev, U = np.linalg.eigh(Cm @ Cm.T); U = U[:, ::-1][:, :cfg['K']]; ev = ev[::-1][:cfg['K']]
        V = (Cm.T @ U) / np.sqrt(ev)
        xA, xQ = (XA.mean(0) @ V) @ V.T, (XQ.mean(0) @ V) @ V.T
    beta = np.linalg.solve(K + alpha * np.trace(K) / n * np.eye(n), (YA - rt) - c * xA)
    out = rt + c * xQ + (FQ @ FA.T) @ beta
    if cfg['clip']:
        vis = np.concatenate([Y[[row_of[(s, d)] for d in common]] for s in src] + [YA])
        out = np.clip(out, vis.min(0), vis.max(0))
    return out
