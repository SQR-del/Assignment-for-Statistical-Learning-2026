"""nn_v2 variant 'base_I1_I2' with the tuned settings K=32, epochs=800, dropout=0.2 (everything else as 09_nn_base_I1_I2)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main, SEED

TUNED = dict(K=32, epochs=800, dropout=0.2)


def predict(tab, hidden, qry, seeds):
    from nn_v2 import Config, prepare, ensemble_predict
    cfg = Config(**TUNED); S = prepare(tab, hidden, cfg)
    return ensemble_predict(S, tab, qry, 'base_I1_I2', seeds, SEED, cfg)[0]


if __name__ == '__main__':
    main(predict, __file__)
