"""Reorganise the project to follow the midterm report's sections (2026-11-16).
Plain directory renames, so it works the same on the local clone and on the server. Run from the project root:
    python tools/restructure_moves.py            # dry run: print the moves
    python tools/restructure_moves.py --apply
Paths that do not exist are skipped (the server has extra, untracked folders; the local clone lacks some)."""
import os, sys, glob, shutil

M = []                      # (source, destination)
def mv(a, b): M.append((a, b))

# §2 label generation
mv("Question 2", "2_label_generation")
# §3 validation
mv("methods/harness.py", "3_validation/harness.py")
mv("code/validate.py", "3_validation/validate.py")
mv("results/validation", "3_validation/results")
# §5 official scores
mv("results/kaggle_scores.csv", "5_official_scores/kaggle_scores.csv")
# shared code
mv("code", "src")
# §4 models, grouped as in the report
CAT = {"baselines": ["00_zero", "01_target_mean", "02_null_median", "03_compound_mean", "04_null_plus_shift"],
       "anchor_transfer": ["05a_krr_local", "05b_krr_joint", "06_nn_base", "07_nn_base_I1", "08_nn_base_I2", "09_nn_base_I1_I2",
                           "10_blend", "11_nn_v2_base", "12_nn_v2_base_I1", "13_nn_v2_base_I2", "14_nn_v2_base_I1_I2",
                           "15_blend_v2", "nn_v2_lab"],
       "embedding": ["16_embedding", "17_embedding_dsig", "18_smiles_cnn", "19_blend_emb", "emb_lab"],
       "classical": ["20_pca_ridge", "21_mlp_sklearn", "22_mlp_wide", "23_ensemble_r1", "24_blend_r1", "classic_lab"]}
for cat, names in CAT.items():
    for n in names: mv("methods/" + n, "4_models/%s/%s" % (cat, n))
mv("methods/README.md", "4_models/README.md")
mv("results/nn_v2_lab", "4_models/anchor_transfer/nn_v2_lab/runs")
# history -> archive
for d in ["blend_final", "blend_v1", "krr_joint_study", "krr_null_v1", "krr_v2", "nn_base_I2_v1", "nn_base_I1_I2_v1",
          "nn_base_v1_posthoc"]:
    mv("results/" + d, "archive/old_experiments/" + d)
mv("results/methods", "archive/old_experiments/methods_kaggle_json")     # local clone: only the old per-method kaggle_score.json
mv("results/README.md", "archive/old_experiments/README.md")
mv("submissions/README.md", "archive/old_experiments/submissions_README.md")
mv("submissions/methods/README.md", "archive/old_experiments/submissions_methods_README.md")
mv("submissions/krr_null_v1_summary.json", "archive/old_experiments/krr_null_v1/krr_null_v1_summary.json")
# reports
mv("开题报告", "reports/proposal")
mv("中期报告", "reports/midterm")

apply = "--apply" in sys.argv
server = os.path.exists("data")          # on the server the big export folder results/methods must stay where it is
for a, b in M:
    if a == "results/methods" and server:
        print("keep  ", a, "(server exports)"); continue
    if not os.path.exists(a):
        print("skip  ", a); continue
    if os.path.exists(b):
        print("EXISTS", b); sys.exit(1)
    print("move  ", a, "->", b)
    if apply:
        os.makedirs(os.path.dirname(b) or ".", exist_ok=True)
        shutil.move(a, b)
if apply:
    for d in ["methods", "submissions/methods", "submissions", "results"]:
        if os.path.isdir(d) and not os.listdir(d): os.rmdir(d); print("rmdir ", d)
