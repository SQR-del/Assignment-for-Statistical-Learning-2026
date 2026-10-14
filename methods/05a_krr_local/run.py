"""Local KRR (局部 KRR, version 2 of code/krr.py): KRR v1 + projection of the source shift on 64 components + clipping; c=0.4, alpha=3."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main, krr


def predict(tab, hidden, qry, seeds):
    return krr(tab, hidden, qry, 2)


if __name__ == '__main__':
    main(predict, __file__)
