"""复现分支自己报告的分数：在分支的协议 P1 下运行移植后的模型。
P1：去掉全部阳性对照；轮流取一种 T/NK 细胞为目标，训练 = 其余三种 T/NK 的全部行 + 目标的 15 个锚点（不含 B/髓系），
测试 = 目标的全部非锚点药；预测截断到训练范围；分数 = 4 折 MRRMSE 的平均。
分支报告：embedding 0.899（5 个种子），SMILES-CNN 0.904，hybrid 0.937（各 3 个种子），融合约 0.90。
用法: CUDA_VISIBLE_DEVICES=0 python repro_p1.py [--seeds 3]     输出: results/emb_lab/repro_p1.csv
"""
import os, sys, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import harness
from harness import PROJ
from common import Table, rms
from emb_adapter import predict_models

p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, default=3); a = p.parse_args()
tab = Table(); rows = []
for o in tab.TNK:
    refs = [c for c in tab.TNK if c != o]
    vis = np.flatnonzero(~tab.ctrl & (np.isin(tab.c_row, refs) | ((tab.c_row == o) & np.isin(tab.d_row, tab.ANC))))
    qry = np.flatnonzero(~tab.ctrl & (tab.c_row == o) & ~np.isin(tab.d_row, tab.ANC))
    P = predict_models(tab, vis, qry, a.seeds); P["zero"] = np.zeros_like(P["embed"])
    for k, v in P.items():
        rows.append(dict(target=tab.cells[o], model=k, n_train=len(vis), n_test=len(qry), mrrmse=float(rms(v - tab.Y[qry]).mean())))
    print(tab.cells[o], {k: round(float(rms(v - tab.Y[qry]).mean()), 4) for k, v in P.items()}, flush=True)
R = pd.DataFrame(rows); out = os.path.join(PROJ, "results", "emb_lab"); os.makedirs(out, exist_ok=True)
R.to_csv(os.path.join(out, "repro_p1.csv"), index=False)
print(R.pivot(index="target", columns="model", values="mrrmse").round(4).to_string())
print("4 折平均:", R.groupby("model").mrrmse.mean().round(4).to_dict())
