"""在组员分支的协议 P1 下评估本仓库的方法（00-15），与 repro_p1.py 的口径相同：
去掉全部阳性对照；轮流取一种 T/NK 细胞为目标，训练 = 其余三种 T/NK 的全部行 + 目标的 15 个锚点（不含 B/髓系），
测试 = 目标的全部非锚点药；分数 = 4 折 MRRMSE 的平均（含 CD8）。不在训练集里的行在交给方法前置为 NaN。
用法: CUDA_VISIBLE_DEVICES=0 python p1_ours.py [--seeds 3]     输出: results/emb_lab/p1_ours.csv
"""
import os, sys, glob, argparse, importlib.util, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE); sys.path.insert(0, M)
import harness
from harness import PROJ
from common import Table, rms

p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, default=3); a = p.parse_args()
tab = Table(); rows = []
folders = sorted(f for f in glob.glob(os.path.join(M, "[01][0-9]*_*")) if os.path.isdir(f) and int(os.path.basename(f)[:2]) <= 15)
for f in folders:
    name = os.path.basename(f)
    spec = importlib.util.spec_from_file_location("m_" + name, os.path.join(f, "run.py")); mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        for o in tab.TNK:
            refs = [c for c in tab.TNK if c != o]
            vis = ~tab.ctrl & (np.isin(tab.c_row, refs) | ((tab.c_row == o) & np.isin(tab.d_row, tab.ANC)))
            hid = np.flatnonzero(~vis); qry = np.flatnonzero(~tab.ctrl & (tab.c_row == o) & ~np.isin(tab.d_row, tab.ANC))
            pred = mod.predict(harness.masked(tab, hid), hid, qry, a.seeds)
            assert np.isfinite(pred).all(), "non-finite prediction"
            rows.append(dict(method=name, target=tab.cells[o], mrrmse=float(rms(pred - tab.Y[qry]).mean())))
        print(name, round(np.mean([r["mrrmse"] for r in rows if r["method"] == name]), 4), flush=True)
    except Exception:
        print(name, "FAILED"); traceback.print_exc()
R = pd.DataFrame(rows); out = os.path.join(PROJ, "results", "emb_lab"); R.to_csv(os.path.join(out, "p1_ours.csv"), index=False)
print(R.pivot(index="method", columns="target", values="mrrmse").assign(P1=lambda x: x.mean(1)).round(4).to_string())
