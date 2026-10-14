import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
from models import CLASSIC_METHODS

_FAMILY = ['krr_rbf', 'krr_linear_sig', 'krr_rbf_sig', 'pg_anchor_ridge']

def predict(ctx, method='krr_linear_sig', **params):
    fn, defaults = CLASSIC_METHODS[method]
    return fn(ctx['view'], **{**defaults, **params})

if __name__ == '__main__':
    main(predict, __file__, dict(method='krr_linear_sig'))
