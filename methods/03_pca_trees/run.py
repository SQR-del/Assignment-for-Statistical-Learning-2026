import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
from models import CLASSIC_METHODS

_FAMILY = ['pca_ridge', 'pca_xgb', 'pca_rf', 'pca_mlp', 'xgb_multi', 'random_forest', 'extra_trees', 'hgb_multi']

def predict(ctx, method='pca_ridge', **params):
    fn, defaults = CLASSIC_METHODS[method]
    return fn(ctx['view'], **{**defaults, **params})

if __name__ == '__main__':
    main(predict, __file__, dict(method='pca_ridge'))
