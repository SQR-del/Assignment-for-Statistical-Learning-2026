"""Baseline: per-gene mean of all visible training rows of the target cell type (anchors and controls)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from harness import main, by_target, visible_rows


def predict(tab, hidden, qry, seeds):
    out = np.zeros((len(qry), tab.G))
    for t, pos in by_target(tab, qry):
        out[pos] = tab.Y[visible_rows(tab, t, hidden)].mean(0)
    return out


if __name__ == '__main__':
    main(predict, __file__)
