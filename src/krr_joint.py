"""Joint KRR (联合 KRR): kernel ridge over all cell types, used as a control experiment against the base network.

Local KRR (krr.py):  one model per target cell type, labels = its 15 anchors.
Joint KRR (this file): one model for all cell types, labels = all visible rows.

Same as nn.py 'base' / 'base_I1': the same prepare() (TSVD of the visible rows, leave-one-out target encodings of cell
type and compound, optional analog encoding), the same training rows (all visible rows of all cell types), the same
input standardisation and the same output space (K TSVD coordinates, mapped back with V). Only the learner differs:
kernel ridge regression instead of the MLP ensemble.

kind : 'linear' (= ridge regression on the encodings) or 'rbf' (bandwidth gamma0 / median squared distance)
loss : 'mse'    closed-form kernel ridge (squared error on the TSVD coordinates)
       'mrrmse' the network's loss, mean over rows of sqrt((||z - f||^2 + out-of-subspace residual) / G),
                solved by iteratively re-weighted kernel ridge (weights 1 / row RMSE, normalised to mean 1)
lam  : ridge penalty relative to the average kernel diagonal, lambda = lam * tr(K) / n
"""
import numpy as np


def features(S, use_ate):
    import torch
    return torch.cat([S['cte'], S['dte']] + ([S['ate']] if use_ate else []), 1).double().cpu().numpy()


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0)


class KRRJoint:
    """Fit once per fold; predict for several penalties without refitting the kernel."""

    def __init__(self, S, G, kind='rbf', gamma0=1.0, use_ate=False):
        tr = np.flatnonzero(S['vis']); self.tr, self.G, self.kind = tr, G, kind
        X = features(S, use_ate); X = X / (X[tr].std(0) + 1e-6)          # same standardisation as the network
        Z = S['z'].double().cpu().numpy(); self.res = S['res'].double().cpu().numpy()[tr]
        self.zbar = Z[tr].mean(0); self.Y = Z[tr] - self.zbar
        self.X, self.Xt = X, X[tr]
        if kind == 'rbf':
            D = sqdist(self.Xt, self.Xt); self.gamma = gamma0 / np.median(D[np.triu_indices(len(tr), 1)])
            self.K = np.exp(-self.gamma * D)
        else:
            self.K = self.Xt @ self.Xt.T
        self.scale = np.trace(self.K) / len(tr)
        self.V = S['V'].double().cpu().numpy()

    def kq(self, qry):
        Xq = self.X[qry]
        return np.exp(-self.gamma * sqdist(Xq, self.Xt)) if self.kind == 'rbf' else Xq @ self.Xt.T

    def coef(self, lam, loss='mse', iters=15):
        l = lam * self.scale; w = np.ones(len(self.tr))
        for _ in range(iters if loss == 'mrrmse' else 1):
            a = np.linalg.solve(self.K + l * np.diag(1.0 / w), self.Y)
            if loss != 'mrrmse':
                break
            rm = np.sqrt((((self.Y - self.K @ a) ** 2).sum(1) + self.res) / self.G)
            w = 1.0 / np.maximum(rm, 1e-6); w /= w.mean()
        return a

    def predict(self, qry, lam, loss='mse'):
        """Predictions in gene space for rows qry."""
        return (self.zbar + self.kq(qry) @ self.coef(lam, loss)) @ self.V.T
