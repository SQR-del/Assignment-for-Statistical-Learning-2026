import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
from models import CLASSIC_METHODS

_FAMILY = ['rrr']

def predict(ctx, method='rrr', **params):
    fn, defaults = CLASSIC_METHODS[method]
    return fn(ctx['view'], **{**defaults, **params})

if __name__ == '__main__':
    main(predict, __file__, dict(method='rrr'))
