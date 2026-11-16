"""Update path references in the READMEs after the restructure (2026-11-16). Run from the project root.
archive/old_experiments/ is left as it was (historical record, written for the old layout)."""
import io, os, re, glob

CAT = {}
for cat, names in {"baselines": "00_zero 01_target_mean 02_null_median 03_compound_mean 04_null_plus_shift",
                   "anchor_transfer": "05a_krr_local 05b_krr_joint 06_nn_base 07_nn_base_I1 08_nn_base_I2 09_nn_base_I1_I2 10_blend "
                                      "11_nn_v2_base 12_nn_v2_base_I1 13_nn_v2_base_I2 14_nn_v2_base_I1_I2 15_blend_v2 nn_v2_lab",
                   "embedding": "16_embedding 17_embedding_dsig 18_smiles_cnn 19_blend_emb emb_lab",
                   "classical": "20_pca_ridge 21_mlp_sklearn 22_mlp_wide 23_ensemble_r1 24_blend_r1 classic_lab"}.items():
    for n in names.split(): CAT[n] = cat

PAIRS = [("Question 2/", "2_label_generation/"), ("`Question 2`", "`2_label_generation`"),
         ("results/nn_v2_lab", "4_models/anchor_transfer/nn_v2_lab/runs"),
         ("results/kaggle_scores.csv", "5_official_scores/kaggle_scores.csv"),
         ("results/validation", "3_validation/results"),
         ("methods/harness.py", "3_validation/harness.py"), ("methods/README.md", "4_models/README.md"),
         ("code/validate.py", "3_validation/validate.py"), ("`code/", "`src/"), (" code/", " src/"), ("（code/", "（src/")]
for d in ["blend_final", "blend_v1", "krr_joint_study", "krr_null_v1", "krr_v2", "nn_base_I2_v1", "nn_base_I1_I2_v1", "nn_base_v1_posthoc"]:
    PAIRS.append(("results/" + d, "archive/old_experiments/" + d))
METHOD = re.compile(r"(?<![\w/])methods/(%s)" % "|".join(sorted(CAT, key=len, reverse=True)))

files = [p for p in glob.glob("**/*.md", recursive=True) if not p.replace("\\", "/").startswith(("archive/old_experiments/", "reports/"))]
for p in files:
    s0 = s = io.open(p, encoding="utf-8").read()
    for a, b in PAIRS: s = s.replace(a, b)
    s = METHOD.sub(lambda m: "4_models/%s/%s" % (CAT[m.group(1)], m.group(1)), s)
    if re.match(r"4_models[\\/][a-z_]+[\\/]\d", p):                 # method folders: their ../README.md was methods/README.md
        s = s.replace("`../README.md`", "`4_models/README.md`")
    if s != s0:
        io.open(p, "w", encoding="utf-8", newline="\n").write(s); print("updated", p)
