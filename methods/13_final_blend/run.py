import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
import numpy as np
import nn
from sklearn.decomposition import PCA
from models import m_ensemble

# Fixed-weight blend of the round-2 winners (full data, fixed weights).
W = dict(embed=0.50, smiles=0.30, hybrid=0.20)

def predict(ctx, seeds_embed=5, seeds_other=3):
    s = ctx['split']; v = ctx['view']; n_cells, n_drugs = len(ctx['cells']), len(ctx['drugs'])
    emb = np.median(np.stack([nn.train_embed(s['x_tr'], ctx['ytr'], s['x_te'], n_cells, n_drugs,
               emb=128, n_dim=25, seed=20251001 + r) for r in range(seeds_embed)]), 0)
    smi = np.median(np.stack([nn.train_smiles(s['x_tr'], s['sym_tr'], ctx['ytr'], s['x_te'],
               s['sym_te'], ctx['vocab'], n_cells, n_drugs, n_dim=25, seed=20251001 + r)
               for r in range(seeds_other)]), 0)
    pd_ = PCA(n_components=64, random_state=20251001).fit(v['Dsig_train'])
    Xtr = pd_.transform(v['Dsig_train']).astype('float32'); Xte = pd_.transform(v['Dsig_test']).astype('float32')
    hyb = np.median(np.stack([nn.train_hybrid(s['x_tr'], Xtr, ctx['ytr'], s['x_te'], Xte, n_cells,
               n_drugs, emb=128, n_dim=25, seed=20251001 + r) for r in range(seeds_other)]), 0)
    return W['embed'] * emb + W['smiles'] * smi + W['hybrid'] * hyb

if __name__ == '__main__':
    main(predict, __file__, dict(seeds_embed=5, seeds_other=3))
