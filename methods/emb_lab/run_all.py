"""在本仓库的内部验证（methods/harness.py 的 34 折）上运行分支的三个网络及其融合，每折只训练一次。
用法: CUDA_VISIBLE_DEVICES=0 python run_all.py --part 0 --nparts 3 [--seeds 3]     （按折号取模分给几块 GPU）
       python run_all.py --merge                                                      （合并各部分，写入 methods/16-19）
输出: results/emb_lab/rows_<模型>_part<k>.csv、preds/（各折的预测，供以后做融合）
"""
import os, sys, json, glob, argparse, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE); sys.path.insert(0, M)
import harness
from harness import PROJ
from common import Table, rms
OUT = os.path.join(PROJ, "results", "emb_lab")
FOLDER = dict(embed="16_embedding", hybrid="17_embedding_dsig", smiles="18_smiles_cnn", blend_b="19_blend_emb")

p = argparse.ArgumentParser(); p.add_argument("--part", type=int, default=0); p.add_argument("--nparts", type=int, default=1)
p.add_argument("--seeds", type=int, default=3); p.add_argument("--merge", action="store_true"); a = p.parse_args()
os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
if a.merge:
    for k, f in FOLDER.items():
        R = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(os.path.join(OUT, "rows_%s_part*.csv" % k)))])
        assert R.name.nunique() == 34, R.name.nunique()
        d = os.path.join(M, f); os.makedirs(d, exist_ok=True); R.to_csv(os.path.join(d, "rows.csv"), index=False)
        s = harness.summarize(R); s["seeds"] = a.seeds
        json.dump(s, open(os.path.join(d, "scores.json"), "w"), indent=2); harness.attach_kaggle(d)
        print(f, s["objective"], s["A"], s["V1w"], s["V1"])
    sys.exit()
from emb_adapter import predict_models
tab = Table(); recs = {k: [] for k in FOLDER}; t0 = time.time()
for k, (proto, name, hid, qry) in enumerate(harness.folds(tab)):
    if k % a.nparts != a.part: continue
    tm = harness.masked(tab, hid); hs = set(hid.tolist()); vis = [i for i in range(tab.N) if i not in hs]
    P = predict_models(tm, vis, qry, a.seeds)
    for m, v in P.items():
        assert np.isfinite(v).all()
        recs[m] += [dict(proto=proto, name=name, row=int(i), err=float(e)) for i, e in zip(qry, rms(v - tab.Y[qry]))]
    np.savez_compressed(os.path.join(OUT, "preds", "fold%02d.npz" % k), qry=qry, **{m: v.astype(np.float32) for m, v in P.items()})
    print("fold %d/34 %s %s [t=%.0fs]" % (k + 1, proto, name[:40], time.time() - t0), flush=True)
for m, r in recs.items(): pd.DataFrame(r).to_csv(os.path.join(OUT, "rows_%s_part%d.csv" % (m, a.part)), index=False)
print("DONE")
