import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
from models import CLASSIC_METHODS

_FAMILY = ['linear', 'ridge', 'bayesridge', 'pls', 'knn', 'ridge_resid', 'knn_resid']

def predict(ctx, method='ridge', **params):
    fn, defaults = CLASSIC_METHODS[method]
    return fn(ctx['view'], **{**defaults, **params})

if __name__ == '__main__':
    main(predict, __file__, dict(method='ridge'))
