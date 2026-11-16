"""Fix the cross-folder paths after tools/restructure_moves.py (2026-11-16). Run from the project root, same on the
local clone and on the server. Every literal replacement must match the expected number of times, else it stops."""
import io, os, glob, sys

def rd(p): return io.open(p, encoding="utf-8").read()
def wr(p, s): io.open(p, "w", encoding="utf-8", newline="\n").write(s)
def rep(p, pairs):
    s = rd(p)
    for a, b, *n in pairs:
        n = n[0] if n else 1
        c = s.count(a)
        assert c == n, (p, a[:70], c, n)
        s = s.replace(a, b)
    wr(p, s); print("patched", p)

# Folder-independent way to find the project root: walk up until a folder containing src/ is found.
FIND = ('_R = os.path.dirname(os.path.abspath(__file__))\n'
        'while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)\n'
        'sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]')

# 1. every method run.py
OLD_M = "import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
OLD_A = ('import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname('
         'os.path.dirname(os.path.abspath(__file__))))), "methods"))')
for p in sorted(glob.glob("4_models/*/*/run.py")):
    rep(p, [(OLD_M, "import os, sys\n" + FIND)])
for p in sorted(glob.glob("archive/methods_removed/*/run.py")):
    rep(p, [(OLD_A, "import os, sys\n" + FIND)])

# 2. validation framework
rep("3_validation/harness.py", [
    ("Protocols are those of code/validate.py:", "Protocols are those of 3_validation/validate.py:"),
    ("(from results/kaggle_scores.csv)", "(from 5_official_scores/kaggle_scores.csv)"),
    ("sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'code'))", "sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))"),
    ("(latest valid submission in results/kaggle_scores.csv)", "(latest valid submission in 5_official_scores/kaggle_scores.csv)"),
    ("rec = os.path.join(PROJ, 'results', 'kaggle_scores.csv')", "rec = os.path.join(PROJ, '5_official_scores', 'kaggle_scores.csv')"),
    ("for f in sorted(glob.glob(os.path.join(HERE, '*', 'scores.json'))", "for f in sorted(glob.glob(os.path.join(PROJ, '4_models', '*', '*', 'scores.json'))"),
])
rep("3_validation/validate.py", [
    ("import os, time, json, argparse\n",
     "import os, sys, time, json, argparse\nsys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))\n"),
    ("p.add_argument('--out', default=os.path.join(PROJ, 'results', 'validation'))",
     "p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))"),
])
rep("src/predict.py", [
    ("os.path.join(PROJ, 'results', 'blend_v1')", "os.path.join(PROJ, 'archive', 'old_experiments', 'blend_v1')"),
    ("os.path.join(PROJ, 'results', 'nn_base_I2_v1'", "os.path.join(PROJ, 'archive', 'old_experiments', 'nn_base_I2_v1'"),
    ("os.path.join(PROJ, 'results', 'nn_base_I1_I2_v1'", "os.path.join(PROJ, 'archive', 'old_experiments', 'nn_base_I1_I2_v1'"),
    ("os.path.join(PROJ, 'results', 'krr_v2'", "os.path.join(PROJ, 'archive', 'old_experiments', 'krr_v2'"),
])

# 3. labs
L = "4_models/anchor_transfer/nn_v2_lab/"
OLD_L = "HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))"
NEW_L = "HERE = os.path.dirname(os.path.abspath(__file__))\n" + FIND
for f in ["compare.py", "run_aligned.py", "run_blend.py", "run_export.py", "run_variant.py"]:
    pairs = [(OLD_L, NEW_L)]
    if f == "compare.py": pairs.append(('L = os.path.join(PROJ, "results", "nn_v2_lab")', 'L = os.path.join(HERE, "runs")'))
    if f == "run_variant.py": pairs.append(('out = os.path.join(PROJ, "results", "nn_v2_lab", a.name)', 'out = os.path.join(HERE, "runs", a.name)'))
    if f == "run_blend.py": pairs.append(('out = os.path.join(PROJ, "results", "nn_v2_lab", "blend_" + a.name)', 'out = os.path.join(HERE, "runs", "blend_" + a.name)'))
    if f == "run_aligned.py":
        pairs += [('Q2 = os.path.join(PROJ, "Question 2")', 'Q2 = os.path.join(PROJ, "2_label_generation")'),
                  ('out = os.path.join(PROJ, "results", "nn_v2_lab", "al_%s_%s" % (a.mode, a.name))', 'out = os.path.join(HERE, "runs", "al_%s_%s" % (a.mode, a.name))'),
                  ("（Question 2/sim/", "（2_label_generation/sim/")]
    rep(L + f, pairs)
OLD_E = "HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE); sys.path.insert(0, M)"
NEW_E = "HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE)\n" + FIND
for p in ["4_models/embedding/emb_lab/run_all.py", "4_models/embedding/emb_lab/p1_ours.py", "4_models/classical/classic_lab/run_all.py"]:
    rep(p, [(OLD_E, NEW_E)])
rep("4_models/embedding/emb_lab/repro_p1.py", [(OLD_L, NEW_L)])
rep("4_models/classical/classic_lab/export_branch_flow.py", [
    ("HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))",
     "HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE\nwhile not os.path.isdir(os.path.join(ROOT, \"src\")): ROOT = os.path.dirname(ROOT)"),
    ('sys.path.insert(0, os.path.join(ROOT, "code"))', 'sys.path.insert(0, os.path.join(ROOT, "src"))'),
])

# 4. label generation: the folder name is written into the scripts (also in absolute server paths)
for p in sorted(glob.glob("2_label_generation/*.py") + glob.glob("2_label_generation/*.R") + glob.glob("2_label_generation/*.sh")):
    s = rd(p)
    if "Question 2" in s:
        wr(p, s.replace("Question 2", "2_label_generation")); print("renamed refs in", p)
for p in ["2_label_generation/12_krr_aligned.py", "2_label_generation/12b_krr_aligned_strong.py"]:
    rep(p, [('sys.path.insert(0, os.path.join(PROJ, "methods")); sys.path.insert(0, HERE)',
             'sys.path[:0] = [os.path.join(PROJ, "src"), os.path.join(PROJ, "3_validation"), HERE]')])

# 5. report figure scripts
for p in ["reports/midterm/make_fig.py", "reports/midterm/make_fig_noise.py"]:
    if os.path.exists(p):
        s = rd(p); n = s.count("../Question 2/out")
        assert n >= 1, p
        wr(p, s.replace("../Question 2/out", "../../2_label_generation/out")); print("patched", p)
print("ALL DONE")
