"""比较 nn_v2 各实验与参照。用法: python compare.py [参照名=ref] [名称前缀过滤]
差值 = 实验 - 参照（负数为更好）；区间为对被预测行重抽样的 95% 区间。"""
import os, sys, json, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
from harness import PROJ
L = os.path.join(HERE, "runs"); ref = sys.argv[1] if len(sys.argv) > 1 else "ref"; pre = sys.argv[2] if len(sys.argv) > 2 else ""
R0 = pd.read_csv(os.path.join(L, ref, "rows.csv")); s0 = json.load(open(os.path.join(L, ref, "scores.json"))); rng = np.random.default_rng(0)
def ci(x):
    bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]; return "%+.4f [%+.4f,%+.4f]" % (x.mean(), np.quantile(bs, .025), np.quantile(bs, .975))
print("%-22s %9s %8s %8s %8s | %-26s %-26s %-26s" % ("实验", "objective", "A", "V1w", "V1", "A 行差", "V1w 行差", "V1 行差"))
print("%-22s %9.4f %8.4f %8.4f %8.4f" % (ref + "(参照)", s0["objective"], s0["A"], s0["V1w"], s0["V1"]))
for d in sorted(glob.glob(os.path.join(L, "*"))):
    n = os.path.basename(d)
    if n == ref or not n.startswith(pre) or not os.path.exists(os.path.join(d, "scores.json")): continue
    s = json.load(open(os.path.join(d, "scores.json"))); M = R0.merge(pd.read_csv(os.path.join(d, "rows.csv")), on=["proto", "name", "row"], suffixes=("_0", "_1")); M["d"] = M.err_1 - M.err_0
    a = M[M.proto == "A"].d.to_numpy(); v = M[M.proto == "V1"]; w = v[v.name.str.endswith("|weak")].d.to_numpy()
    print("%-22s %9.4f %8.4f %8.4f %8.4f | %-26s %-26s %-26s" % (n, s["objective"], s["A"], s["V1w"], s["V1"], ci(a), ci(w), ci(v.d.to_numpy())))
