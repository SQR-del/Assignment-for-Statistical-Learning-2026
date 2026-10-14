"""Classic (numpy / scikit-learn) model zoo for the perturbation benchmark.

Every model is a callable  fn(view, seed=SEED, **params) -> y_pred_test
(n_test x n_genes, float32).  `view` is the dict produced by common.build_views.
All models are fitted ONLY on the training fold.  Necessary hyper-parameters are
function arguments so the same model can be tuned by keyword.

Torch models (embedding MLP, SMILES-CNN, autoencoder, bilinear CP, full-output
MLP) live in `nn.py`.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb

SEED = 20251001
GENE_SUBSET = None  # optional list of gene indices for expensive tree models


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _std_targets(y_train):
    mu = y_train.mean(0)
    sd = y_train.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def _scaled(view):
    sc = StandardScaler().fit(view["Xc_train"])
    return sc.transform(view["Xc_train"]), sc.transform(view["Xc_test"])


def _ridge_pca(view, alpha, n_comp=128):
    """Ridge on target-space PCA components (per-gene standardised)."""
    mu, sd = _std_targets(view["y_train"])
    Yz = (view["y_train"] - mu) / sd
    k = min(n_comp, Yz.shape[0] - 1, Yz.shape[1])
    pca = PCA(n_components=k, random_state=SEED).fit(Yz)
    Xtr, Xte = _scaled(view)
    m = Ridge(alpha=alpha).fit(Xtr, pca.transform(Yz))
    return (pca.inverse_transform(m.predict(Xte)) * sd + mu).astype(np.float32)


# --------------------------------------------------------------------------- #
# trivial baselines
# --------------------------------------------------------------------------- #
def m_global_mean(view, seed=SEED):
    return np.repeat(view["y_train"].mean(0)[None, :], len(view["y_test"]), 0)


def m_cell_anchor_mean(view, seed=SEED):
    """= the leakage-safe cell-type signature (csig)."""
    return view["Csig_test"].copy()


def m_drug_mean(view, seed=SEED):
    """= the leave-own-cell-type-out drug signature (dsig)."""
    return view["Dsig_test"].copy()


def m_additive(view, seed=SEED):
    g = view["y_train"].mean(0)
    return (view["Dsig_test"] + view["Csig_test"] - g).astype(np.float32)


# --------------------------------------------------------------------------- #
# linear / neighbor models on compact features
# --------------------------------------------------------------------------- #
def m_linear(view, seed=SEED):
    Xtr, Xte = _scaled(view)
    return LinearRegression().fit(Xtr, view["y_train"]).predict(Xte)


def m_ridge(view, seed=SEED, alpha=10.0):
    Xtr, Xte = _scaled(view)
    return Ridge(alpha=alpha).fit(Xtr, view["y_train"]).predict(Xte)


def m_bayesridge(view, seed=SEED):
    Xtr, Xte = _scaled(view)
    return BayesianRidge(max_iter=300).fit(Xtr, view["y_train"]).predict(Xte)


def m_pls(view, seed=SEED, n_components=20):
    Xtr, Xte = _scaled(view)
    return PLSRegression(n_components=n_components, scale=False).fit(Xtr, view["y_train"]).predict(Xte)


def m_knn(view, seed=SEED, k=5):
    Xtr, Xte = _scaled(view)
    return KNeighborsRegressor(n_neighbors=k, weights="distance").fit(Xtr, view["y_train"]).predict(Xte)


def m_krr_rbf(view, seed=SEED, alpha=1.0):
    Xtr, Xte = _scaled(view)
    return KernelRidge(kernel="rbf", alpha=alpha, gamma=1.0 / Xtr.shape[1]).fit(Xtr, view["y_train"]).predict(Xte)


# --------------------------------------------------------------------------- #
# kernel ridge on high-dimensional signature features
# --------------------------------------------------------------------------- #
def _sig_features(view):
    return (np.concatenate([view["Dsig_train"], view["Csig_train"]], 1),
            np.concatenate([view["Dsig_test"], view["Csig_test"]], 1))


def m_krr_linear_sig(view, seed=SEED, alpha=1.0):
    Xtr, Xte = _sig_features(view)
    return KernelRidge(kernel="linear", alpha=alpha).fit(Xtr, view["y_train"]).predict(Xte)


def m_krr_rbf_sig(view, seed=SEED, alpha=1.0):
    Xtr, Xte = _sig_features(view)
    return KernelRidge(kernel="rbf", alpha=alpha, gamma=1.0 / Xtr.shape[1]).fit(Xtr, view["y_train"]).predict(Xte)


# --------------------------------------------------------------------------- #
# per-gene anchor transfer: learn a cell-specific map ref-signature -> target
# --------------------------------------------------------------------------- #
def m_pg_anchor_ridge(view, seed=SEED, use_csig=False, lam=1e-2):
    """For every gene g fit a tiny ridge from the target cell's 15 anchor drugs
    mapping the reference signature -> target response, then apply to test drugs.
    Vectorised over genes.  Falls back to dsig in P2 (no anchors)."""
    mask = view["anchor_mask_train"]
    if mask.sum() == 0:
        return view["Dsig_test"].copy()
    Dtr, Dte = view["Dsig_train"][mask], view["Dsig_test"]
    Ytr = view["y_train"][mask]
    cols_tr, cols_te = [Dtr], [Dte]
    if use_csig:
        cols_tr.append(view["Csig_train"][mask]); cols_te.append(view["Csig_test"])
    Xtr = np.stack(cols_tr, -1).astype(np.float64)
    Xte = np.stack(cols_te, -1).astype(np.float64)
    Y = Ytr.astype(np.float64)
    mu = Xtr.mean(0, keepdims=True)
    Xc = Xtr - mu
    Yc = Y - Y.mean(0, keepdims=True)
    k = Xc.shape[-1]
    XtX = np.einsum("ngk,ngl->gkl", Xc, Xc) + lam * np.eye(k)
    XtY = np.einsum("ngk,ng->gk", Xc, Yc)
    beta = np.linalg.solve(XtX, XtY[..., None])[..., 0]
    intercept = Y.mean(0) - np.einsum("gk,gk->g", beta, mu[0])
    return (np.einsum("ngk,gk->ng", (Xte - mu), beta) + intercept).astype(np.float32)


def m_pg_anchor_affine(view, seed=SEED, lam=1e-2):
    return m_pg_anchor_ridge(view, use_csig=True, lam=lam)


# --------------------------------------------------------------------------- #
# tree ensembles
# --------------------------------------------------------------------------- #
def m_extra_trees(view, seed=SEED, n_estimators=100, max_depth=6):
    Xtr, Xte = _scaled(view)
    m = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth,
                            random_state=seed, n_jobs=-1)
    return m.fit(Xtr, view["y_train"]).predict(Xte)


def m_random_forest(view, seed=SEED, n_estimators=100, max_depth=6):
    Xtr, Xte = _scaled(view)
    m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                              random_state=seed, n_jobs=-1)
    return m.fit(Xtr, view["y_train"]).predict(Xte)


def m_xgb_multi(view, seed=SEED, n_estimators=150, max_depth=4, learning_rate=0.1):
    Xtr, Xte = _scaled(view)
    m = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=learning_rate, subsample=0.9, colsample_bytree=0.8,
                         reg_lambda=1.0, multi_strategy="multi_output_tree",
                         tree_method="hist", random_state=seed, n_jobs=8, verbosity=0)
    return m.fit(Xtr, view["y_train"]).predict(Xte)


def m_hgb_multi(view, seed=SEED, max_iter=100, max_depth=3, learning_rate=0.1, n_genes=500):
    """Gradient boosting on the top-variance gene subset (MultiOutput is costly)."""
    sub = (np.argsort(view["y_train"].var(0))[::-1][:n_genes] if GENE_SUBSET is None else GENE_SUBSET)
    Xtr, Xte = _scaled(view)
    m = MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate,
        random_state=seed), n_jobs=-1)
    m.fit(Xtr, view["y_train"][:, sub])
    pred = view["Dsig_test"].copy()
    pred[:, sub] = m.predict(Xte)
    return pred


# --------------------------------------------------------------------------- #
# sklearn MLP on compact features (no torch dependency)
# --------------------------------------------------------------------------- #
def m_mlp_sklearn(view, seed=SEED, hidden=(256, 128), alpha=1e-3, max_iter=300,
                  batch_size=64, lr=1e-3, early_stopping=True):
    Xtr, Xte = _scaled(view)
    mu, sd = _std_targets(view["y_train"])
    Ytr = (view["y_train"] - mu) / sd
    m = MLPRegressor(hidden_layer_sizes=tuple(hidden), activation="relu", alpha=alpha,
                     batch_size=batch_size, learning_rate_init=lr, max_iter=max_iter,
                     early_stopping=early_stopping, n_iter_no_change=15, random_state=seed)
    m.fit(Xtr, Ytr)
    return (m.predict(Xte) * sd + mu).astype(np.float32)


# --------------------------------------------------------------------------- #
# target-space PCA compression (makes tree / non-linear base learners tractable)
# --------------------------------------------------------------------------- #
def _pca_base(view, base="ridge", n_comp=128, **kw):
    mu, sd = _std_targets(view["y_train"])
    Yz = (view["y_train"] - mu) / sd
    k = min(n_comp, Yz.shape[0] - 1, Yz.shape[1])
    pca = PCA(n_components=k, random_state=SEED).fit(Yz)
    Ztr = pca.transform(Yz)
    Xtr, Xte = _scaled(view)
    if base == "ridge":
        est = Ridge(alpha=kw.get("alpha", 10.0))
    elif base == "xgb":
        est = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.08,
                               subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                               multi_strategy="multi_output_tree", tree_method="hist",
                               random_state=SEED, n_jobs=8, verbosity=0)
    elif base == "rf":
        est = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=SEED, n_jobs=-1)
    elif base == "mlp":
        est = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=64,
                           learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                           n_iter_no_change=15, random_state=SEED)
    else:
        raise ValueError(base)
    est.fit(Xtr, Ztr)
    return (pca.inverse_transform(est.predict(Xte)) * sd + mu).astype(np.float32)


def m_pca_ridge(view, seed=SEED, n_comp=128, alpha=10.0):
    return _pca_base(view, "ridge", n_comp, alpha=alpha)


def m_pca_xgb(view, seed=SEED, n_comp=128):
    return _pca_base(view, "xgb", n_comp)


def m_pca_rf(view, seed=SEED, n_comp=128):
    return _pca_base(view, "rf", n_comp)


def m_pca_mlp(view, seed=SEED, n_comp=128):
    return _pca_base(view, "mlp", n_comp)


# --------------------------------------------------------------------------- #
# residual ("transfer") variants: predict y - dsig
# --------------------------------------------------------------------------- #
def m_ridge_resid(view, seed=SEED, alpha=100.0):
    Xtr, Xte = _scaled(view)
    R = view["y_train"] - view["Dsig_train"]
    return (Ridge(alpha=alpha).fit(Xtr, R).predict(Xte) + view["Dsig_test"]).astype(np.float32)


def m_knn_resid(view, seed=SEED, k=5):
    Xtr, Xte = _scaled(view)
    R = view["y_train"] - view["Dsig_train"]
    m = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(Xtr, R)
    return (m.predict(Xte) + view["Dsig_test"]).astype(np.float32)


# --------------------------------------------------------------------------- #
# reduced-rank regression (supervised low-rank output)
# --------------------------------------------------------------------------- #
def m_rrr(view, seed=SEED, rank=20, alpha=10.0):
    """Y = X B with rank(B) <= rank: ridge coefficient matrix projected onto its
    leading right-singular subspace.  Supervised alternative to output-PCA."""
    Xtr, Xte = _scaled(view)
    ymean = view["y_train"].mean(0)
    Yc = view["y_train"] - ymean
    B = Ridge(alpha=alpha, fit_intercept=False).fit(Xtr, Yc).coef_.T
    M = Xtr @ B
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    Vr = Vt[:rank].T
    B_rrr = B @ Vr @ Vr.T
    return (ymean + Xte @ B_rrr).astype(np.float32)


# --------------------------------------------------------------------------- #
# robust ensemble of diverse linear/kernel families (parameters = tuned config)
# --------------------------------------------------------------------------- #
def m_ensemble(view, seed=SEED, ridge_alpha=100.0, knn_k=10, pls_n=10, n_comp=128):
    """Equal-weight mean of 5 decorrelated families.  Defaults are the config
    chosen by nested CV in REPORT.md (ridge_alpha=100, knn_k=10, pls_n=10)."""
    Xtr, Xte = _scaled(view)
    Xs_tr, Xs_te = _sig_features(view)
    parts = [
        _ridge_pca(view, ridge_alpha, n_comp),
        Ridge(alpha=ridge_alpha).fit(Xtr, view["y_train"]).predict(Xte),
        PLSRegression(n_components=pls_n, scale=False).fit(Xtr, view["y_train"]).predict(Xte),
        KNeighborsRegressor(n_neighbors=knn_k, weights="distance").fit(Xtr, view["y_train"]).predict(Xte),
        KernelRidge(kernel="linear", alpha=1.0).fit(Xs_tr, view["y_train"]).predict(Xs_te),
    ]
    return np.mean(parts, 0).astype(np.float32)


# --------------------------------------------------------------------------- #
# registry: name -> (callable, default params) for the classic families
# --------------------------------------------------------------------------- #
CLASSIC_METHODS = {
    "global_mean": (m_global_mean, {}),
    "cell_anchor_mean": (m_cell_anchor_mean, {}),
    "drug_mean": (m_drug_mean, {}),
    "additive": (m_additive, {}),
    "linear": (m_linear, {}),
    "ridge": (m_ridge, dict(alpha=10.0)),
    "bayesridge": (m_bayesridge, {}),
    "pls": (m_pls, dict(n_components=20)),
    "knn": (m_knn, dict(k=5)),
    "krr_rbf": (m_krr_rbf, dict(alpha=1.0)),
    "krr_linear_sig": (m_krr_linear_sig, dict(alpha=1.0)),
    "krr_rbf_sig": (m_krr_rbf_sig, dict(alpha=1.0)),
    "pg_anchor_ridge": (m_pg_anchor_ridge, {}),
    "extra_trees": (m_extra_trees, {}),
    "random_forest": (m_random_forest, {}),
    "xgb_multi": (m_xgb_multi, {}),
    "hgb_multi": (m_hgb_multi, {}),
    "mlp_sklearn": (m_mlp_sklearn, dict(hidden=(256, 128), alpha=0.01)),
    "pca_ridge": (m_pca_ridge, dict(alpha=10.0)),
    "pca_xgb": (m_pca_xgb, {}),
    "pca_rf": (m_pca_rf, {}),
    "pca_mlp": (m_pca_mlp, {}),
    "ridge_resid": (m_ridge_resid, dict(alpha=100.0)),
    "knn_resid": (m_knn_resid, dict(k=5)),
    "rrr": (m_rrr, dict(rank=20, alpha=10.0)),
    "ensemble": (m_ensemble, dict(ridge_alpha=100.0, knn_k=10, pls_n=10)),
}
