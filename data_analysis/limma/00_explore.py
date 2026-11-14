"""Quick checks of the raw single-cell table and metadata."""
import os
import json
import duckdb
import pandas as pd

from _common import DATA, OUT, ensure_out

ensure_out()

obs = pd.read_csv(os.path.join(DATA, "adata_obs_meta.csv"))
print("obs", obs.shape)
print("cells per cell_type x control")
print(pd.crosstab(obs.cell_type, obs.control).to_string())
print("cells per donor x cell_type")
print(pd.crosstab(obs.donor_id, obs.cell_type).to_string())
print("dose by control")
print(obs.groupby(["control", "sm_name"]).dose_uM.unique().to_string())

con = duckdb.connect()
con.execute(f"SET temp_directory='{os.path.join(OUT, 'duck_tmp')}'")
counts = f"'{os.path.join(DATA, 'adata_train.parquet')}'"
print(con.execute(f"SELECT count(*) AS rows, count(DISTINCT gene) AS genes, "
                  f"sum(CASE WHEN count=0 THEN 1 ELSE 0 END) AS zeros, "
                  f"min(count) AS min_c, max(count) AS max_c, avg(count) AS mean_c "
                  f"FROM read_parquet({counts})").df().to_string(index=False))
print(con.execute(f"SELECT count(*) AS n FROM read_parquet({counts}) "
                  f"WHERE normalized_count IS NULL").df().to_string(index=False))

de = pd.read_parquet(os.path.join(DATA, "de_train.parquet"))
genes = [c for c in de.columns if c not in
         ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]]
adata_genes = con.execute(f"SELECT DISTINCT gene FROM read_parquet({counts})").df().gene
print("de genes", len(genes), "adata genes", len(adata_genes),
      "overlap", len(set(genes) & set(adata_genes)))
pd.DataFrame({"gene": genes}).to_csv(os.path.join(OUT, "genes.csv"), index=False)

rows_per_cell = con.execute(
    f"SELECT obs_id, count(*) AS n FROM read_parquet({counts}) GROUP BY obs_id"
).df()
print("rows per cell")
print(rows_per_cell.n.describe().to_string())
print("cells with no count>0:", int((rows_per_cell.n == 0).sum()))

dose_by_control = {str(k): [float(x) for x in v] for k, v in
                   obs.groupby(["control", "sm_name"]).dose_uM.unique().items()}
summary = {
    "obs_rows": int(len(obs)),
    "control_cells": {str(k): int(v) for k, v in obs.control.value_counts().items()},
    "dose_by_control": dose_by_control,
    "de_genes": len(genes), "adata_genes": int(len(adata_genes)),
    "gene_overlap": len(set(genes) & set(adata_genes)),
}
with open(os.path.join(OUT, "explore_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("saved", OUT)
