"""nn_v2 实验：在 base_I2 上逐项打开开关，用与其他方法相同的内部验证（methods/harness.py）评估。
用法: CUDA_VISIBLE_DEVICES=3 python run_variant.py <名称> '<Config 覆盖项的 JSON>' [--variant base_I2] [--seeds 3] [--base-seed 20260930]
输出: results/nn_v2_lab/<名称>/rows.csv, scores.json
例:   python run_variant.py ref '{}'            # 开关全关，应与 08_nn_base_I2 完全相同
      python run_variant.py K32 '{"K": 32}'
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
import harness
from harness import PROJ


def main():
    p = argparse.ArgumentParser(); p.add_argument("name"); p.add_argument("overrides"); p.add_argument("--variant", default="base_I2")
    p.add_argument("--seeds", type=int, default=3); p.add_argument("--base-seed", type=int, default=harness.SEED)
    a = p.parse_args(); ov = json.loads(a.overrides)
    from nn_v2 import Config, prepare, ensemble_predict

    def predict(tab, hidden, qry, seeds):
        cfg = Config(**ov); S = prepare(tab, hidden, cfg)
        return ensemble_predict(S, tab, qry, a.variant, seeds, a.base_seed, cfg)[0]

    out = os.path.join(HERE, "runs", a.name); os.makedirs(out, exist_ok=True)
    json.dump(dict(overrides=ov, variant=a.variant, seeds=a.seeds, base_seed=a.base_seed), open(os.path.join(out, "config.json"), "w"))
    harness.evaluate(predict, a.seeds, out)


if __name__ == "__main__":
    main()
