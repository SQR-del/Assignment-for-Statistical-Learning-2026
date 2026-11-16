"""4th-place post-processing (shrink part, M = 1) applied to the base network: p * min(mean|p|^0.6, 1) per row."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
import numpy as np
from harness import main, nn


def predict(tab, hidden, qry, seeds):
    p = nn(tab, hidden, qry, 'base', seeds)
    return p * np.minimum(np.abs(p).mean(1, keepdims=True) ** 0.6, 1.0)


if __name__ == '__main__':
    main(predict, __file__)
