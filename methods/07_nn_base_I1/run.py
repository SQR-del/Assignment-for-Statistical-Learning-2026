"""Base network + I1 (analog target encoding), code/nn.py variant 'base_I1'."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main, nn


def predict(tab, hidden, qry, seeds):
    return nn(tab, hidden, qry, 'base_I1', seeds)


if __name__ == '__main__':
    main(predict, __file__)
