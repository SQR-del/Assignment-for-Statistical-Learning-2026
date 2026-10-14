"""Original LGT (latent gated transfer), variant L3 with its selected config, run through the archived code.
yhat = target anchor mean + per-source gates on the query's source rows, split into a PCA subspace (block-wise gates)
and its complement (one gate per source)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "methods"))
import numpy as np
from harness import main, by_target
from common import PROJ
sys.path.insert(0, os.path.join(PROJ, 'archive', 'code_lgt_20260930'))
from lgt import prepare_fold, fit  # noqa: E402

CONFIG = dict(variant='L3', k=4, blocks=4, la=1.0, ld=1.0, lb=0.01)   # selected_final_configs.json of lgt_20260930_v1


def predict(tab, hidden, qry, seeds):
    E, P = len(tab.cells), len(tab.comps); hs = set(np.asarray(hidden).tolist())
    y = np.full((E, P, tab.G), np.nan); obs = np.zeros((E, P), bool)
    for (c, d), i in tab.row_of.items():                     # LGT excludes positive controls everywhere
        if i not in hs and not tab.ctrl[i]: y[c, d] = tab.Y[i]; obs[c, d] = True
    out = np.zeros((len(qry), tab.G))
    for t, pos in by_target(tab, qry):
        anchors = np.flatnonzero(obs[t]); groups = {}
        for k in pos:                                        # one model per set of available sources (as in LGT)
            d = tab.d_row[qry[k]]
            groups.setdefault(tuple(s for s in tab.TNK if s != t and obs[s, d]), []).append(k)
        for sources, ks in groups.items():
            mask = np.zeros_like(obs); mask[list(sources)] = obs[list(sources)]; mask[t, anchors] = obs[t, anchors]
            f = prepare_fold(y, mask, t, list(sources), anchors, max_k=32, seed=20260930)
            out[ks] = fit(f, CONFIG).predict(y, mask, [tab.d_row[qry[k]] for k in ks])
    return out


if __name__ == '__main__':
    main(predict, __file__)
