"""Base network (4th-place style, code/nn.py variant 'base'): leave-one-out target encodings on TSVD components -> MLP ensemble."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main, nn


def predict(tab, hidden, qry, seeds):
    return nn(tab, hidden, qry, 'base', seeds)


if __name__ == '__main__':
    main(predict, __file__)
