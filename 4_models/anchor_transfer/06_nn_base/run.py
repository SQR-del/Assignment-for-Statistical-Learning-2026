"""Base network (4th-place style, code/nn.py variant 'base'): leave-one-out target encodings on TSVD components -> MLP ensemble."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
from harness import main, nn


def predict(tab, hidden, qry, seeds):
    return nn(tab, hidden, qry, 'base', seeds)


if __name__ == '__main__':
    main(predict, __file__)
