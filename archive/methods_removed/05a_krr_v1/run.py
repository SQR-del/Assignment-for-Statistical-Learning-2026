"""KRR v1: null-centred anchor-analog kernel ridge regression (code/krr.py, version 1: c=0.2, alpha=10)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "methods"))
from harness import main, krr


def predict(tab, hidden, qry, seeds):
    return krr(tab, hidden, qry, 1)


if __name__ == '__main__':
    main(predict, __file__)
