"""nn_v2 + I2: same as 08_nn_base_I2, but the cell-type target encoding is a leave-one-out median (code/nn_v2.py)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main, SEED


def predict(tab, hidden, qry, seeds):
    from nn_v2 import Config, prepare, ensemble_predict          # same call sequence as harness.nn, other module
    cfg = Config(cte_median='all'); S = prepare(tab, hidden, cfg)
    return ensemble_predict(S, tab, qry, 'base_I2', seeds, SEED, cfg)[0]


if __name__ == '__main__':
    main(predict, __file__)
