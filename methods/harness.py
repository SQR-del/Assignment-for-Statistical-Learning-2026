"""Shared validation / export harness for the method folders.

A method is one function  predict(tab, hidden, qry, seeds) -> array [len(qry), G]
    tab    : common.Table whose `hidden` rows are set to NaN, so a method that reads them returns NaN and fails
    hidden : rows the method must not read;  qry : rows to predict (a subset of hidden)
Protocols are those of code/validate.py:
    A   : each T/NK cell type in turn keeps only its 15 anchors + 2 controls; predict its official test compounds.
          A15 = A without CD8+.
    V1  : B / Myeloid leave-one-anchor-out; V1w = anchors that are weak in the sources (like most test compounds).
    objective = mean(V1w / 1.1625, A15 / 0.9894)  (KRR v1 = 1.000; lower is better)

    python run.py            # internal validation -> scores.json (+ official Kaggle scores), rows.csv in the method folder
    python harness.py        # only refresh the Kaggle scores in every scores.json (from results/kaggle_scores.csv)
    python run.py --export   # train on all rows -> results/methods/<method>/<method>_kaggle_ids.csv (Kaggle id order)
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'code'))
from common import PROJ, Table, rms  # noqa: E402

REF_KRR1 = dict(V1w=1.1625, A15=0.9894)
SEED = 20260930


def masked(tab, hidden):
    t = object.__new__(Table); t.__dict__.update(tab.__dict__)
    t.Y = tab.Y.copy(); t.Y[hidden] = np.nan
    return t


def by_target(tab, qry):
    """(target cell index, positions in qry) for every target cell type among the query rows."""
    ct = tab.c_row[qry]
    return [(t, np.flatnonzero(ct == t)) for t in np.unique(ct)]


def visible_rows(tab, t, hidden, controls=True):
    hs = set(np.asarray(hidden).tolist())
    return [i for i in np.flatnonzero(tab.c_row == t) if i not in hs and (controls or not tab.ctrl[i])]


def folds(tab):
    Y, row_of = tab.Y, tab.row_of
    cut = np.quantile([rms(Y[[row_of[(c, d)] for c in tab.SRC]].mean(0)) for d in tab.QOFF], .9)
    out = []
    for o in tab.TNK:
        hid = np.flatnonzero((tab.c_row == o) & ~tab.ctrl & ~np.isin(tab.d_row, tab.ANC))
        out.append(('A', tab.cells[o], hid, np.array([i for i in hid if tab.d_row[i] in tab.QOFF])))
    for t in tab.SPARSE:
        for d in tab.ANC:
            i = row_of[(t, d)]; weak = rms(Y[[row_of[(c, d)] for c in tab.SRC]].mean(0)) < cut
            out.append(('V1', tab.cells[t] + '|' + tab.comps[d] + ('|weak' if weak else ''), np.array([i]), np.array([i])))
    return out


def summarize(R):
    a = R[R.proto == 'A'].groupby('name').err.mean()
    b = R[R.proto == 'V1'].copy(); b['target'] = b['name'].str.split('|').str[0]; b['weak'] = b['name'].str.endswith('|weak')
    s = dict(A=a.mean(), A15=a.drop('T cells CD8+').mean(), V1=b.groupby('target').err.mean().mean(),
             V1w=b[b.weak].groupby('target').err.mean().mean())
    s['objective'] = (s['V1w'] / REF_KRR1['V1w'] + s['A15'] / REF_KRR1['A15']) / 2
    s.update({f'A_{k}': v for k, v in a.items()})
    return {k: round(float(v), 4) for k, v in s.items()}


def evaluate(predict, seeds, out_dir):
    tab = Table(); Y = tab.Y; recs = []; t0 = time.time()
    for k, (proto, name, hid, qry) in enumerate(folds(tab)):
        pred = predict(masked(tab, hid), hid, qry, seeds)
        assert pred.shape == (len(qry), tab.G) and np.isfinite(pred).all(), 'non-finite prediction (hidden rows read?)'
        recs += [dict(proto=proto, name=name, row=int(i), err=float(e)) for i, e in zip(qry, rms(pred - Y[qry]))]
        print(f'fold {k + 1}/34 {proto} {name[:40]} [t={time.time() - t0:.0f}s]', flush=True)
    R = pd.DataFrame(recs); R.to_csv(os.path.join(out_dir, 'rows.csv'), index=False)
    s = summarize(R); s['seeds'] = seeds
    json.dump(s, open(os.path.join(out_dir, 'scores.json'), 'w'), indent=2)
    attach_kaggle(out_dir)
    print(open(os.path.join(out_dir, 'scores.json'), encoding='utf-8').read())


def attach_kaggle(folder):
    """Add the official Kaggle scores of this method (latest valid submission in results/kaggle_scores.csv) to
    scores.json. Kaggle scores are post-hoc evaluations of fixed methods and are never used for model selection."""
    path = os.path.join(folder, 'scores.json'); s = json.load(open(path, encoding='utf-8'))
    for k in [k for k in s if k.startswith('kaggle_')]: del s[k]
    name = os.path.basename(os.path.normpath(folder)); rec = os.path.join(PROJ, 'results', 'kaggle_scores.csv')
    K = pd.read_csv(rec, encoding='utf-8-sig') if os.path.exists(rec) else pd.DataFrame(columns=['method_folder'])
    K = K[K.method_folder.astype(str).map(lambda f: f == name or f.endswith('/' + name))]
    K = K[K.valid.astype(str) == 'True']
    if len(K):
        r = K.iloc[-1]
        s.update(kaggle_private=float(r.private), kaggle_public=float(r.public), kaggle_file=r.kaggle_file,
                 kaggle_submitted=r.submitted, kaggle_precision=r.precision)
    else:
        s.update(kaggle_private=None, kaggle_public=None, kaggle_file='未提交')
    json.dump(s, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    return s


def export(predict, seeds, name):
    import hashlib
    tab0 = Table(); tab, qry = tab0.with_queries(np.nan)
    pred = predict(tab, qry, qry, seeds)
    out = os.path.join(PROJ, 'results', 'methods', name); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'{name}_kaggle_ids.csv'); tab0.write_submission(pred, path)
    print('wrote', path, 'sha256', hashlib.sha256(open(path, 'rb').read()).hexdigest())


def main(predict, file):
    p = argparse.ArgumentParser()
    p.add_argument('--export', action='store_true')
    p.add_argument('--seeds', type=int, default=None, help='network seeds (default 3 for validation, 10 for export)')
    a = p.parse_args()
    seeds = a.seeds or (10 if a.export else 3)
    folder = os.path.dirname(os.path.abspath(file)); name = os.path.basename(folder)
    export(predict, seeds, name) if a.export else evaluate(predict, seeds, folder)


# ---------------- helpers shared by several methods ----------------
def krr(tab, hidden, qry, version):
    """KRR v1/v2 per target: sources = NK/CD4+/Treg minus the target, anchors = the target's visible anchors."""
    from krr import krr_predict
    hs = set(np.asarray(hidden).tolist()); out = np.zeros((len(qry), tab.G))
    for t, pos in by_target(tab, qry):
        src = [s for s in tab.SRC if s != t]
        anc = [a for a in tab.ANC if (t, a) in tab.row_of and tab.row_of[(t, a)] not in hs]
        out[pos] = krr_predict(tab.Y, tab.row_of, len(tab.comps), t, src, anc, list(tab.d_row[qry[pos]]), version,
                               skip=set(tab.CTRLC))
    return out


def nn(tab, hidden, qry, variant, seeds, S=None):
    """Network variant ('base', 'base_I1', 'base_I2', 'base_I1_I2'); S = prepared encodings (reused if given)."""
    from nn import Config, prepare, ensemble_predict
    cfg = Config(); S = S if S is not None else prepare(tab, hidden, cfg)
    return ensemble_predict(S, tab, qry, variant, seeds, SEED, cfg)[0]


if __name__ == '__main__':      # python harness.py : refresh the Kaggle fields of every scores.json without re-running
    import glob
    for f in sorted(glob.glob(os.path.join(HERE, '*', 'scores.json'))
                    + glob.glob(os.path.join(PROJ, 'archive', 'methods_removed', '*', 'scores.json'))):
        s = attach_kaggle(os.path.dirname(f))
        print(f'{os.path.basename(os.path.dirname(f)):20s} internal objective {s["objective"]}   '
              f'Kaggle private / public {s["kaggle_private"]} / {s["kaggle_public"]}')
