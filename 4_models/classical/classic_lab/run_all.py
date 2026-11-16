"""在本仓库的内部验证（methods/harness.py 的 34 折）上运行组员分支第一轮提交过 Kaggle 的模型（分支 REPORT.md §12）。
每折只构造一次分支的特征（build_views），同一折上的所有模型共用。
用法: CUDA_VISIBLE_DEVICES=0 python3 run_all.py --part 0 --nparts 4
       python3 run_all.py --merge                       （合并各部分，写 results/classic_lab/summary.csv 与方法文件夹）
"""
import os, sys, json, glob, argparse, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE)
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
import harness
from harness import PROJ
from common import Table, rms
OUT = os.path.join(PROJ, "results", "classic_lab")
FOLDER = dict(pca_ridge="20_pca_ridge", mlp_sklearn="21_mlp_sklearn", mlp_wide="22_mlp_wide",
              ensemble_r1="23_ensemble_r1", blend_r1="24_blend_r1")

p = argparse.ArgumentParser()
p.add_argument("--part", type=int, default=0); p.add_argument("--nparts", type=int, default=1)
p.add_argument("--folds", default=None, help="comma-separated fold numbers (timing tests)")
p.add_argument("--tag", default=None, help="output tag for --folds runs (default: test)")
p.add_argument("--merge", action="store_true"); a = p.parse_args()
os.makedirs(OUT, exist_ok=True)

if a.merge:
    from classic_adapter import ALL
    rows, summ = {}, []
    for m in ALL:
        fs = sorted(glob.glob(os.path.join(OUT, "rows_%s_part*.csv" % m)))
        if not fs: print("missing", m); continue
        R = pd.concat([pd.read_csv(x) for x in fs]).drop_duplicates(["proto", "name", "row"])
        if R.name.nunique() != 34: print("incomplete", m, R.name.nunique()); continue
        s = harness.summarize(R); summ.append(dict(model=m, **s)); rows[m] = R
    S = pd.DataFrame(summ).sort_values("objective"); S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(S[["model", "A", "V1w", "V1", "objective"]].round(4).to_string(index=False))
    for m, f in FOLDER.items():
        if m not in rows: continue
        d = os.path.join(M, f); os.makedirs(d, exist_ok=True); rows[m].to_csv(os.path.join(d, "rows.csv"), index=False)
        s = harness.summarize(rows[m]); s["seeds"] = 1
        json.dump(s, open(os.path.join(d, "scores.json"), "w"), indent=2); harness.attach_kaggle(d)
    sys.exit()

from classic_adapter import ALL, predict_models
which = ALL
tab = Table(); recs = {m: [] for m in which}; t0 = time.time()
sel = set(int(x) for x in a.folds.split(",")) if a.folds else None
for k, (proto, name, hid, qry) in enumerate(harness.folds(tab)):
    if (sel is not None and k not in sel) or (sel is None and k % a.nparts != a.part): continue
    tm = harness.masked(tab, hid); hs = set(hid.tolist()); vis = [i for i in range(tab.N) if i not in hs]
    P = predict_models(tm, vis, qry, which)
    for m, v in P.items():
        assert np.isfinite(v).all(), m
        recs[m] += [dict(proto=proto, name=name, row=int(i), err=float(e)) for i, e in zip(qry, rms(v - tab.Y[qry]))]
    print("fold %d/34 %s %s [t=%.0fs]" % (k + 1, proto, name[:40], time.time() - t0), flush=True)
tag = (a.tag or "test") if sel is not None else "part%d" % a.part
for m, r in recs.items(): pd.DataFrame(r).to_csv(os.path.join(OUT, "rows_%s_%s.csv" % (m, tag)), index=False)
print("DONE")
