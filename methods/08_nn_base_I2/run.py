"""Base network + I2 (magnitude head), code/nn.py variant 'base_I2'."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main, nn


def predict(tab, hidden, qry, seeds):
    return nn(tab, hidden, qry, 'base_I2', seeds)


if __name__ == '__main__':
    main(predict, __file__)
