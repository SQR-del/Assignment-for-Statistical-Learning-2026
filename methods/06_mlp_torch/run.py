import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
import numpy as np
import models as M
import nn

def predict(ctx, hidden=(512, 256), epochs=150, wd=1e-4, residual=False, **kw):
    Xtr, Xte = M._scaled(ctx['view'])
    return nn.train_full_mlp(Xtr, Xte, ctx['ytr'], hidden=hidden, epochs=epochs, wd=wd,
                             residual=residual,
                             dsig_tr=ctx['view']['Dsig_train'], dsig_te=ctx['view']['Dsig_test'])

if __name__ == '__main__':
    main(predict, __file__, dict(hidden=(512, 256), epochs=150))
