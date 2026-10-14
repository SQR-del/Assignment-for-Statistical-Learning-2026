import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
import numpy as np
import nn

def predict(ctx, R=20, steps=1200, lr=0.01, wd=1e-5):
    df = ctx['df']; fold = ctx['fold']; cid = ctx['cid']; did = ctx['did']
    c_tr = [cid[df.at[i, 'cell_type']] for i in fold.train_idx]
    d_tr = [did[df.at[i, 'drug']] for i in fold.train_idx]
    model = nn.fit_cp(c_tr, d_tr, ctx['ytr'], len(ctx['cells']), len(ctx['drugs']),
                      R=R, steps=steps, lr=lr, wd=wd, seed=20251001)
    c_te = [cid[c] for c, _ in ctx['view']['rows_test']]
    d_te = [did[d] for _, d in ctx['view']['rows_test']]
    return nn.predict_cp(model, c_te, d_te)

if __name__ == '__main__':
    main(predict, __file__, dict(R=20, steps=1200))
