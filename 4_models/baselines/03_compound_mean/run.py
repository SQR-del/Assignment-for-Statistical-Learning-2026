"""Baseline: mean response of the same compound in the other T/NK cell types (where observed and visible)."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
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
