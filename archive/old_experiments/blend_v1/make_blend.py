"""Final blend v1 (weights fixed by the pre-declared internal selection + confirmation replicates):
   0.4 * (base + magnitude head) + 0.4 * (base + analog TE + magnitude head) + 0.2 * KRR v2. All inputs in Kaggle id order."""
import hashlib, json, os, shutil
import numpy as np, pandas as pd
R = '/home/sqr/NIPS 2023 Single-Cell Perturbation Prediction/results'
parts = [(0.4, f'{R}/nn_base_I2_v1/base_I2_kaggle_ids.csv'), (0.4, f'{R}/nn_base_I1_I2_v1/base_I1_I2_kaggle_ids.csv'),
         (0.2, f'{R}/krr_v2/krr_v2_kaggle_ids.csv')]
frames = [pd.read_csv(p) for _, p in parts]
cols = list(frames[0].columns)
for f in frames:
    assert list(f.columns) == cols and list(f.id) == list(range(255)) and np.isfinite(f[cols[1:]].to_numpy()).all()
pred = sum(w * f[cols[1:]].to_numpy() for (w, _), f in zip(parts, frames))
out = f'{R}/blend_v1'; os.makedirs(out, exist_ok=False)
fr = pd.DataFrame(pred, columns=cols[1:]); fr.insert(0, 'id', range(255))
path = f'{out}/blend_v1_kaggle_ids.csv'; fr.to_csv(path, index=False, float_format='%.9g')
back = pd.read_csv(path); assert list(back.columns) == cols
np.testing.assert_allclose(back[cols[1:]].to_numpy(), pred, rtol=1e-6, atol=1e-6)
rms = lambda x: np.sqrt((x ** 2).mean(1))
info = dict(weights=[w for w, _ in parts], components=[dict(path=p, sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest()) for _, p in parts],
            internal=dict(selection='pre-declared objective mean(V1w/1.1625, A15/0.9894); confirmation on 2 independent replicates',
                          blend_objective=[0.9254, 0.9253], reference_base_I2=[0.9360, 0.9358]),
            pred_row_RMS={'B cells': float(rms(pred[:128]).mean()), 'Myeloid cells': float(rms(pred[128:]).mean())},
            sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest())
json.dump(info, open(f'{out}/summary.json', 'w'), indent=2)
shutil.copy(__file__, f'{out}/make_blend.py')
print(json.dumps(info, indent=1))
