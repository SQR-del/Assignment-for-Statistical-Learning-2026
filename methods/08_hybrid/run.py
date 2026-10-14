import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
import numpy as np
from sklearn.decomposition import PCA
import nn

def predict(ctx, emb=128, hidden=(256, 512, 1024), n_dim=25, epochs=400, lr=3e-3,
            drop=0.2, n_extra=64, reps=3):
    s = ctx['split']; v = ctx['view']
    pd_ = PCA(n_components=n_extra, random_state=20251001).fit(v['Dsig_train'])
    Xtr = pd_.transform(v['Dsig_train']).astype('float32')
    Xte = pd_.transform(v['Dsig_test']).astype('float32')
    ps = [nn.train_hybrid(s['x_tr'], Xtr, ctx['ytr'], s['x_te'], Xte, len(ctx['cells']),
                          len(ctx['drugs']), emb=emb, hidden=hidden, n_dim=n_dim, epochs=epochs,
                          lr=lr, drop=drop, seed=20251001 + r) for r in range(reps)]
    return np.median(np.stack(ps), 0)

if __name__ == '__main__':
    main(predict, __file__, dict(emb=128, n_dim=25, reps=3))
