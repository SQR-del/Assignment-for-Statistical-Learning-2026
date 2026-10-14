import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
import numpy as np
import nn

def predict(ctx, emb=128, hidden=(256, 512, 1024), n_dim=25, epochs=400, lr=3e-3,
            drop=0.2, reduction='svd', loss='rowrmse', reps=5):
    s = ctx['split']
    ps = [nn.train_embed(s['x_tr'], ctx['ytr'], s['x_te'], len(ctx['cells']), len(ctx['drugs']),
                         emb=emb, hidden=hidden, n_dim=n_dim, epochs=epochs, lr=lr, drop=drop,
                         reduction=reduction, loss=loss, seed=20251001 + r) for r in range(reps)]
    return np.median(np.stack(ps), 0)

if __name__ == '__main__':
    main(predict, __file__, dict(emb=128, n_dim=25, epochs=400, lr=3e-3, reps=5))
