import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
from models import CLASSIC_METHODS

_FAMILY = ['global_mean', 'drug_mean', 'cell_anchor_mean', 'additive']

def predict(ctx, method='drug_mean', **params):
    fn, defaults = CLASSIC_METHODS[method]
    return fn(ctx['view'], **{**defaults, **params})

if __name__ == '__main__':
    main(predict, __file__, dict(method='drug_mean'))
