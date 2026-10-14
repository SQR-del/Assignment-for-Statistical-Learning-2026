import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
import nn

def predict(ctx, k=25, hidden=256, epochs=150, standardize=False):
    s = ctx['split']
    return nn.train_ae_target(s['x_tr'], ctx['ytr'], s['x_te'], len(ctx['cells']),
                              len(ctx['drugs']), k=k, hidden=hidden, epochs=epochs,
                              standardize=standardize, seed=20251001)

if __name__ == '__main__':
    main(predict, __file__, dict(k=25, hidden=256))
