"""Baseline: mean response of the same compound in the other T/NK cell types (where observed and visible)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from harness import main


def predict(tab, hidden, qry, seeds):
    hs = set(np.asarray(hidden).tolist()); out = np.zeros((len(qry), tab.G))
    for k, i in enumerate(qry):
        t, d = tab.c_row[i], tab.d_row[i]
        rows = [tab.row_of[(s, d)] for s in tab.TNK if s != t and (s, d) in tab.row_of and tab.row_of[(s, d)] not in hs]
        out[k] = tab.Y[rows].mean(0)
    return out


if __name__ == '__main__':
    main(predict, __file__)
