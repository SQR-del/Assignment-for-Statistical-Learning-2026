"""对齐验证：协议 A 中，被当作新细胞类型的 T/NK 细胞，其可见的 15 个锚点 + 2 个阳性对照的标签
换成"只用这 17 种药 + DMSO 拟合"的小模型标签（2_label_generation/sim/<细胞>/small_r00，官方流程），
被预测的行仍用官方标签（大模型）。这样锚点与目标的尺度关系与真实任务（B/髓系）一致。
用法: CUDA_VISIBLE_DEVICES=3 python run_aligned.py <名称> '<Config 覆盖项 JSON>' --mode small|calib [--seeds 3]
  small  锚点用小模型标签，不做任何校正（现有模型在真实任务中的处境）
  calib  锚点先按自由度换算到大模型尺度（标签 -> t -> 标签），再交给模型
V1（B/髓系留一）各折不受影响，结果与未对齐时相同；比较时只看协议 A。
输出: results/nn_v2_lab/al_<mode>_<名称>/
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
import harness
from harness import PROJ
Q2 = os.path.join(PROJ, "2_label_generation"); sys.path.insert(0, Q2)
from q2lib import slp_from_t, tag, NG


def small_labels(ct):
    d = os.path.join(Q2, "sim", tag(ct)); comp = pd.read_csv(d + "/small_r00/compounds.csv")
    A = np.fromfile(d + "/small_r00/slp.f64", dtype="<f8").reshape(len(comp), NG)
    return dict(zip(comp.sm_name, A)), float(pd.read_csv(d + "/small_r00/scalars.csv").df_total[0]), float(pd.read_csv(d + "/full/scalars.csv").df_total[0])


def df_map(y, d1, d2):
    t = stats.t.isf(np.clip(10.0 ** (-np.abs(y)) / 2, 1e-300, 0.5), d1)
    return np.sign(y) * np.abs(slp_from_t(t, d2))


def main():
    p = argparse.ArgumentParser(); p.add_argument("name"); p.add_argument("overrides"); p.add_argument("--mode", required=True, choices=["small", "calib"])
    p.add_argument("--variant", default="base_I2"); p.add_argument("--seeds", type=int, default=3); p.add_argument("--base-seed", type=int, default=harness.SEED)
    a = p.parse_args(); ov = json.loads(a.overrides)
    from nn_v2 import Config, prepare, ensemble_predict
    cache = {}

    def predict(tab, hidden, qry, seeds):
        t = int(tab.c_row[qry[0]]); ct = tab.cells[t]
        if t in tab.TNK:                                   # 协议 A：替换目标细胞可见行的标签
            if ct not in cache: cache[ct] = small_labels(ct)
            lab, d1, d2 = cache[ct]; hs = set(np.asarray(hidden).tolist()); n = 0
            for i in np.flatnonzero(tab.c_row == t):
                if i in hs: continue
                y = lab[tab.comps[tab.d_row[i]]]; tab.Y[i] = df_map(y, d1, d2) if a.mode == "calib" else y; n += 1
            assert n == 17 or ct == "T cells CD8+", n
        cfg = Config(**ov); S = prepare(tab, hidden, cfg)
        return ensemble_predict(S, tab, qry, a.variant, seeds, a.base_seed, cfg)[0]

    out = os.path.join(HERE, "runs", "al_%s_%s" % (a.mode, a.name)); os.makedirs(out, exist_ok=True)
    json.dump(dict(overrides=ov, variant=a.variant, seeds=a.seeds, base_seed=a.base_seed, mode=a.mode), open(os.path.join(out, "config.json"), "w"))
    harness.evaluate(predict, a.seeds, out)


if __name__ == "__main__":
    main()
