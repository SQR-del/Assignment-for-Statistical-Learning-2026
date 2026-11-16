"""KRR v1: null-centred anchor-analog kernel ridge regression (code/krr.py, version 1: c=0.2, alpha=10)."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
from harness import main, krr


def predict(tab, hidden, qry, seeds):
    return krr(tab, hidden, qry, 1)


if __name__ == '__main__':
    main(predict, __file__)
