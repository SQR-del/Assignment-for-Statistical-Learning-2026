"""导出 Kaggle 提交文件：用全部训练行训练（10 个随机种子），按 Kaggle 的 id 顺序写出 255 行预测。
用法: CUDA_VISIBLE_DEVICES=3 python run_export.py <base|base_I1|base_I2|base_I1_I2|blend> [--seeds 10]
设置固定为 K=32, epochs=800, dropout=0.2（methods/11–15）。输出: results/methods/v2_<变体>/v2_<变体>_kaggle_ids.csv
测试行在交给模型前标签为 NaN（与 methods/harness.py 的 export 相同）。
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import harness
from harness import krr, SEED
TUNED = dict(K=32, epochs=800, dropout=0.2)


def main():
    p = argparse.ArgumentParser(); p.add_argument("variant"); p.add_argument("--seeds", type=int, default=10); a = p.parse_args()
    from nn_v2 import Config, prepare, ensemble_predict

    def predict(tab, hidden, qry, seeds):
        cfg = Config(**TUNED); S = prepare(tab, hidden, cfg)
        f = lambda v: ensemble_predict(S, tab, qry, v, seeds, SEED, cfg)[0]
        if a.variant == "blend": return 0.4 * f("base_I2") + 0.4 * f("base_I1_I2") + 0.2 * krr(tab, hidden, qry, 2)
        return f(a.variant)

    harness.export(predict, a.seeds, "v2_" + a.variant)


if __name__ == "__main__":
    main()
