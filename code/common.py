"""Shared data handling: training table, index structures, Kaggle id order, metric."""
import os
import numpy as np
import pandas as pd

PROJ = os.environ.get('SCP_PROJ', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
TNK_NAMES = ('NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells')
# CD8+ lacks the Alvocidib / Oprozomib rows, so it is never a KRR source
SOURCE_NAMES = ('NK cells', 'T cells CD4+', 'T regulatory cells')


def rms(x, axis=-1): return np.sqrt(np.mean(np.square(x), axis=axis))


def mrrmse(pred, truth): return float(rms(pred - truth).mean())


class Table:
    """One row per observed (cell type, compound) pair of de_train, labels in Y (signed -log10 p).

    Index conventions: c_row / d_row are integer codes into `cells` / `comps` (both sorted);
    row_of[(c, d)] is the row of a pair.  ANC = the 15 non-control compounds observed in both
    B and Myeloid cells ("anchors"); CTRLC = the 2 positive controls; QOFF = official test compounds.
    """

    def __init__(self, data_dir=None):
        data_dir = data_dir or os.path.join(PROJ, 'data')
        df = pd.read_parquet(os.path.join(data_dir, 'de_train.parquet'))
        self.id_map = pd.read_csv(os.path.join(data_dir, 'id_map.csv'))
        self.genes = [c for c in df.columns if c not in META]; self.G = len(self.genes)
        self.cells = sorted(df.cell_type.unique())
        self.comps = sorted(set(df.sm_name) | set(self.id_map.sm_name))
        self.CI = {c: i for i, c in enumerate(self.cells)}; self.DI = {d: i for i, d in enumerate(self.comps)}
        self.c_row = df.cell_type.map(self.CI).to_numpy(); self.d_row = df.sm_name.map(self.DI).to_numpy()
        self.ctrl = df.control.to_numpy(bool)
        self.Y = df[self.genes].to_numpy(np.float64); self.n_train = len(df)
        self.row_of = {(c, d): i for i, (c, d) in enumerate(zip(self.c_row, self.d_row))}
        self.TNK = [self.CI[c] for c in TNK_NAMES]
        self.SPARSE = [self.CI['B cells'], self.CI['Myeloid cells']]
        self.SRC = [self.CI[c] for c in SOURCE_NAMES]
        self.ANC = sorted(d for d in set(self.d_row[(self.c_row == self.SPARSE[0]) & ~self.ctrl])
                          if (self.SPARSE[1], d) in self.row_of)
        self.CTRLC = sorted(set(self.d_row[self.ctrl])); self.REF = set(self.ANC) | set(self.CTRLC)
        self.QOFF = set(self.DI[s] for s in self.id_map.sm_name)
        # Kaggle submission order: sorted by (cell_type, sm_name), ids 0-127 B cells, 128-254 Myeloid cells.
        # NOTE: data/id_map.csv itself is in a different (interleaved) order and must not be used as-is.
        self.kaggle = self.id_map.sort_values(['cell_type', 'sm_name']).reset_index(drop=True)

    @property
    def N(self): return len(self.Y)

    def with_queries(self, fill=0.0):
        """Copy of the table with the 255 test rows appended (Kaggle id order) and placeholder labels `fill`.
        Returns (table, row indices of the test rows)."""
        t = object.__new__(Table); t.__dict__.update(self.__dict__)
        ec = self.kaggle.cell_type.map(self.CI).to_numpy(); ed = self.kaggle.sm_name.map(self.DI).to_numpy()
        n0 = self.n_train
        t.c_row = np.concatenate([self.c_row[:n0], ec]); t.d_row = np.concatenate([self.d_row[:n0], ed])
        t.ctrl = np.concatenate([self.ctrl[:n0], np.zeros(len(ec), bool)])
        t.Y = np.vstack([self.Y[:n0], np.full((len(ec), self.G), fill)])
        t.row_of = dict(self.row_of)
        for k, (c, d) in enumerate(zip(ec, ed)):
            assert (c, d) not in t.row_of, 'test pair present in training data'
            t.row_of[(c, d)] = n0 + k
        return t, np.arange(n0, n0 + len(ec))

    def write_submission(self, pred, path):
        """Write predictions for the 255 test rows (Kaggle id order) and verify the file round-trips."""
        assert pred.shape == (len(self.kaggle), self.G) and np.isfinite(pred).all()
        fr = pd.DataFrame(pred, columns=self.genes); fr.insert(0, 'id', np.arange(len(self.kaggle)))
        fr.to_csv(path, index=False, float_format='%.9g')
        back = pd.read_csv(path)
        assert list(back.columns) == ['id'] + self.genes and list(back.id) == list(range(len(self.kaggle)))
        np.testing.assert_allclose(back[self.genes].to_numpy(), pred, rtol=1e-6, atol=1e-6)
