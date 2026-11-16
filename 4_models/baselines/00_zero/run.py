"""Baseline: predict 0 for every gene (no differential expression)."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
import numpy as np
from harness import main


def predict(tab, hidden, qry, seeds):
    return np.zeros((len(qry), tab.G))


if __name__ == '__main__':
    main(predict, __file__)
