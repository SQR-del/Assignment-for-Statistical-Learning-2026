"""Baseline: per-gene mean of all visible training rows of the target cell type (anchors and controls)."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
import numpy as np
from harness import main, by_target, visible_rows


def predict(tab, hidden, qry, seeds):
    out = np.zeros((len(qry), tab.G))
    for t, pos in by_target(tab, qry):
        out[pos] = tab.Y[visible_rows(tab, t, hidden)].mean(0)
    return out


if __name__ == '__main__':
    main(predict, __file__)
