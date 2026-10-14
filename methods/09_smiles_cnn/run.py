import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
import numpy as np
import nn

def predict(ctx, n_dim=25, use_drug_emb=True, epochs=400, lr=3e-3, drop=0.2, reps=3):
    s = ctx['split']
    ps = [nn.train_smiles(s['x_tr'], s['sym_tr'], ctx['ytr'], s['x_te'], s['sym_te'],
                          ctx['vocab'], len(ctx['cells']), len(ctx['drugs']), n_dim=n_dim,
                          use_drug_emb=use_drug_emb, epochs=epochs, lr=lr, drop=drop,
                          seed=20251001 + r) for r in range(reps)]
    return np.median(np.stack(ps), 0)

if __name__ == '__main__':
    main(predict, __file__, dict(n_dim=25, use_drug_emb=True, reps=3))
