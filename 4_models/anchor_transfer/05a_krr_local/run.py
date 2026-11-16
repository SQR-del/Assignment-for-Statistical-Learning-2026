"""Local KRR (局部 KRR, version 2 of code/krr.py): KRR v1 + projection of the source shift on 64 components + clipping; c=0.4, alpha=3."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
from harness import main, krr


def predict(tab, hidden, qry, seeds):
    return krr(tab, hidden, qry, 2)


if __name__ == '__main__':
    main(predict, __file__)
