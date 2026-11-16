"""nn_v2 variant 'base_I1' with the tuned settings K=32, epochs=800, dropout=0.2 (everything else as 07_nn_base_I1)."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
from harness import main, SEED

TUNED = dict(K=32, epochs=800, dropout=0.2)


def predict(tab, hidden, qry, seeds):
    from nn_v2 import Config, prepare, ensemble_predict
    cfg = Config(**TUNED); S = prepare(tab, hidden, cfg)
    return ensemble_predict(S, tab, qry, 'base_I1', seeds, SEED, cfg)[0]


if __name__ == '__main__':
    main(predict, __file__)
