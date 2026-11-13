"""blend（0.4 * base_I2 + 0.4 * base_I1_I2 + 0.2 * 局部 KRR），两个网络用 nn_v2 并带 Config 覆盖项。覆盖项为空时等于 methods/10_blend。
用法: CUDA_VISIBLE_DEVICES=3 python run_blend.py <名称> '<Config 覆盖项 JSON>' [--seeds 3] [--base-seed N]     输出: results/nn_v2_lab/blend_<名称>/
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import harness
from harness import PROJ, krr


def main():
    p = argparse.ArgumentParser(); p.add_argument("name"); p.add_argument("overrides"); p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=harness.SEED); a = p.parse_args(); ov = json.loads(a.overrides)
    from nn_v2 import Config, prepare, ensemble_predict

    def predict(tab, hidden, qry, seeds):
        cfg = Config(**ov); S = prepare(tab, hidden, cfg)
        f = lambda v: ensemble_predict(S, tab, qry, v, seeds, a.base_seed, cfg)[0]
        return 0.4 * f("base_I2") + 0.4 * f("base_I1_I2") + 0.2 * krr(tab, hidden, qry, 2)

    out = os.path.join(PROJ, "results", "nn_v2_lab", "blend_" + a.name); os.makedirs(out, exist_ok=True)
    json.dump(dict(overrides=ov, seeds=a.seeds, base_seed=a.base_seed), open(os.path.join(out, "config.json"), "w"))
    harness.evaluate(predict, a.seeds, out)


if __name__ == "__main__":
    main()
