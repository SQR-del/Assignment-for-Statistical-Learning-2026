"""Blend v2: 0.4 * base_I2 + 0.4 * base_I1_I2 + 0.2 * local KRR, networks from nn_v2 with K=32, epochs=800, dropout=0.2.
Same weights and components as 10_blend; only the three network settings differ."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
from harness import main, krr, SEED

TUNED = dict(K=32, epochs=800, dropout=0.2)


def predict(tab, hidden, qry, seeds):
    from nn_v2 import Config, prepare, ensemble_predict
    cfg = Config(**TUNED); S = prepare(tab, hidden, cfg)
    f = lambda v: ensemble_predict(S, tab, qry, v, seeds, SEED, cfg)[0]
    return 0.4 * f('base_I2') + 0.4 * f('base_I1_I2') + 0.2 * krr(tab, hidden, qry, 2)


if __name__ == '__main__':
    main(predict, __file__)
