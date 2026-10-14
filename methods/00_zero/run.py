"""Baseline: predict 0 for every gene (no differential expression)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from harness import main


def predict(tab, hidden, qry, seeds):
    return np.zeros((len(qry), tab.G))


if __name__ == '__main__':
    main(predict, __file__)
