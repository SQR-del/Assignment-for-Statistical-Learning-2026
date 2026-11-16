"""Baseline: null reference of the target + the compound's average shift in the sources.
    yhat = median_t(anchors) + mean_s [ y_s,d - median_s(shared compounds) ],  sources NK / CD4+ / Treg minus target
(= KRR with c = 1 and no anchor-analog term; positive controls excluded from all medians)."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
import numpy as np
from harness import main, by_target, visible_rows


def predict(tab, hidden, qry, seeds):
    out = np.zeros((len(qry), tab.G)); skip = set(tab.CTRLC)
    for t, pos in by_target(tab, qry):
        src = [s for s in tab.SRC if s != t]
        common = [d for d in range(len(tab.comps)) if d not in skip and all((s, d) in tab.row_of for s in src)]
        rS = {s: np.median(tab.Y[[tab.row_of[(s, d)] for d in common]], 0) for s in src}
        rt = np.median(tab.Y[visible_rows(tab, t, hidden, controls=False)], 0)
        for k in pos:
            d = tab.d_row[qry[k]]
            out[k] = rt + np.mean([tab.Y[tab.row_of[(s, d)]] - rS[s] for s in src], 0)
    return out


if __name__ == '__main__':
    main(predict, __file__)
