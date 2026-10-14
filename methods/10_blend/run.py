"""Final model: 0.4 * base_I2 + 0.4 * base_I1_I2 + 0.2 * KRR v2 (same as code/predict.py)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main, krr, nn
from nn import Config, prepare


def predict(tab, hidden, qry, seeds):
    S = prepare(tab, hidden, Config())
    return (0.4 * nn(tab, hidden, qry, 'base_I2', seeds, S) + 0.4 * nn(tab, hidden, qry, 'base_I1_I2', seeds, S)
            + 0.2 * krr(tab, hidden, qry, 2))


if __name__ == '__main__':
    main(predict, __file__)
