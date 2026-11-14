"""Run the complete data-analysis pipeline in order.

Usage:
    python data_analysis/run_all.py            # everything
    python data_analysis/run_all.py --skip-heavy  # skip raw-count pseudobulk

The calibration / error-decomposition steps need the model-side predictions in
``results/predictions/P1__*.npz``; they are skipped with a warning if absent.
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

PIPELINE = [
    ("eda/01_overview.py", []),
    ("eda/02_de_train_analysis.py", []),
    ("limma/00_explore.py", []),
    ("limma/01_pseudobulk.py", ["--variant", "all"]),
    ("limma/02_cell_stats.py", []),
    ("limma/03_cell_limma.py", []),
    ("limma/04_method_comparison.py", []),
    ("limma/05_noise.py", []),
    ("limma/06_calibration.py", []),
    ("limma/07_error_decomp.py", []),
    ("limma/08_figures.py", []),
]

NEEDS_PREDICTIONS = {"limma/06_calibration.py", "limma/07_error_decomp.py",
                     "limma/08_figures.py"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-heavy", action="store_true",
                        help="skip 01_pseudobulk.py (416M-row aggregation)")
    args = parser.parse_args()

    preds = glob.glob(os.path.join(ROOT, "results", "predictions", "*.npz"))
    for rel, extra in PIPELINE:
        if args.skip_heavy and rel == "limma/01_pseudobulk.py":
            print(f"[skip] {rel}")
            continue
        if rel in NEEDS_PREDICTIONS and not preds:
            print(f"[skip] {rel} (no results/predictions/*.npz)")
            continue
        script = os.path.join(HERE, rel)
        print(f"[run ] {rel} {' '.join(extra)}".rstrip(), flush=True)
        subprocess.run([PY, script] + extra, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
