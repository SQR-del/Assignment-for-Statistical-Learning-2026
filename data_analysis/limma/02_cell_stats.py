"""Per-cell logCPM sufficient statistics for cell-level Limma-style DE."""
import os
import duckdb
import numpy as np
import pandas as pd

from _common import DATA, OUT, ensure_out

ensure_out()
COUNT_PARQUET = os.path.join(DATA, "adata_train.parquet")
OBS_CSV = os.path.join(DATA, "adata_obs_meta.csv")

obs = pd.read_csv(OBS_CSV)
key_cols = ["cell_type", "donor_id", "sm_name", "dose_uM", "control"]
keys = obs[key_cols].drop_duplicates().reset_index(drop=True)
keys.insert(0, "group_id", np.arange(len(keys), dtype=np.int32))
keys.to_csv(os.path.join(OUT, "group_keys.csv"), index=False)
obs_map = obs[["obs_id"] + key_cols].merge(keys, on=key_cols, how="left")
genes = pd.read_csv(os.path.join(OUT, "genes.csv"))
gene_map = genes.copy()
gene_map["gene_id"] = np.arange(1, len(gene_map) + 1, dtype=np.int32)

con = duckdb.connect()
con.execute(f"SET temp_directory='{os.path.join(OUT, 'duck_tmp')}'")
con.execute("SET memory_limit='6GB'")
con.execute("SET threads=4")
con.execute("SET preserve_insertion_order=false")
con.execute("SET max_temp_directory_size='18GB'")
con.register("obs_map", obs_map[["obs_id", "group_id"]])
con.register("gene_map", gene_map)

con.execute(f"""
CREATE TABLE lib AS
SELECT obs_id, sum(count)::DOUBLE AS lib
FROM read_parquet('{COUNT_PARQUET}')
GROUP BY obs_id
""")
con.register("lib_df", con.execute("SELECT * FROM lib").df())
print(con.execute("SELECT count(*) AS n, min(lib) AS lo, median(lib) AS med, max(lib) AS hi FROM lib").df().to_string(index=False))

con.execute(f"""
CREATE TABLE cs AS
WITH x AS (
  SELECT o.group_id, g.gene_id,
         log2((c.count + 0.5) / l.lib * 1e6) AS x
  FROM read_parquet('{COUNT_PARQUET}') c
  JOIN obs_map o ON c.obs_id = o.obs_id
  JOIN gene_map g ON c.gene = g.gene
  JOIN lib l ON c.obs_id = l.obs_id
)
SELECT group_id, gene_id, count(*)::INT AS n,
       sum(x) AS sum_x, sum(x * x) AS sum_x2
FROM x
GROUP BY 1, 2
""")
print(con.execute("SELECT count(*) AS rows FROM cs").df().to_string(index=False))
out = os.path.join(OUT, "cell_stats.parquet")
con.execute(f"COPY cs TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
print("saved", out, round(os.path.getsize(out) / 1e6, 1), "MB")
