"""Joint KRR (联合 KRR): kernel ridge on exactly the base network's inputs, rows and output space (code/krr_joint.py).
RBF kernel (gamma0 = 0.25), the network's row-RMSE loss (iteratively re-weighted), lam = 0.003: the best base-input
configuration of the grid in results/krr_joint_study (selected on these same folds, so its score is optimistic)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
from nn import Config, prepare
from krr_joint import KRRJoint


def predict(tab, hidden, qry, seeds):
    S = prepare(tab, hidden, Config())
    return KRRJoint(S, tab.G, kind='rbf', gamma0=0.25, use_ate=False).predict(qry, lam=0.003, loss='mrrmse')


if __name__ == '__main__':
    main(predict, __file__)
