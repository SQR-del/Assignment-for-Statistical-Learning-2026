"""Build pseudobulk tables from the raw single-cell counts.

The original pipeline had four nearly identical scripts (donor / plate /
normalized / split-half).  They are merged here behind ``--variant``:

    donor : donor-level raw counts        -> pseudobulk_long.parquet
    plate : plate-level raw counts        -> pseudobulk_plate_long.parquet
    norm  : donor-level normalized_count  -> normpseudobulk_long.parquet
    half  : donor x split-half counts     -> halfpseudobulk_long.parquet

Run one variant, or ``--variant all`` for the full pipeline:
    python 01_pseudobulk.py --variant all
"""
import argparse
import os

import duckdb
import numpy as np
import pandas as pd

from _common import DATA, OUT, ensure_out

COUNT_PARQUET = os.path.join(DATA, "adata_train.parquet")
OBS_CSV = os.path.join(DATA, "adata_obs_meta.csv")

VARIANTS = {
    "donor": dict(
        keys=["cell_type", "donor_id", "sm_name", "dose_uM", "control"],
        value="count", out="pseudobulk_long.parquet", half=False,
        keys_csv=None),
    "plate": dict(
        keys=["cell_type", "donor_id", "plate_name", "sm_name", "dose_uM",
              "control"],
        value="count", out="pseudobulk_plate_long.parquet", half=False,
        keys_csv="plate_keys.csv"),
    "norm": dict(
        keys=["cell_type", "donor_id", "sm_name", "dose_uM", "control"],
        value="normalized_count", out="normpseudobulk_long.parquet",
        half=False, keys_csv=None),
    "half": dict(
        keys=["cell_type", "donor_id", "sm_name", "dose_uM", "control"],
        value="count", out="halfpseudobulk_long.parquet", half=True,
        keys_csv=None),
}


def build(variant, obs, gene_map):
    spec = VARIANTS[variant]
    keys = obs[spec["keys"]].drop_duplicates().reset_index(drop=True)
    keys.insert(0, "group_id", np.arange(len(keys), dtype=np.int32))
    obs_map = obs[["obs_id"] + spec["keys"]].merge(
        keys, on=spec["keys"], how="left")
    if spec["keys_csv"]:
        keys.to_csv(os.path.join(OUT, spec["keys_csv"]), index=False)

    con = duckdb.connect()
    con.execute(f"SET temp_directory='{os.path.join(OUT, 'duck_tmp')}'")
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET max_temp_directory_size='18GB'")
    con.register("gene_map", gene_map)

    if spec["half"]:
        con.execute("CREATE TABLE om AS SELECT * FROM obs_map")
        con.execute("ALTER TABLE om ADD COLUMN half INTEGER")
        con.execute("UPDATE om SET half = (hash(obs_id) % 2)")
        group_cols, extra = "o.group_id, o.half, g.gene_id", ", p.half"
    else:
        con.execute("CREATE TABLE om AS SELECT * FROM obs_map")
        group_cols, extra = "o.group_id, g.gene_id", ""

    if spec["value"] == "normalized_count":
        agg = f"sum(c.{spec['value']}) AS agg_value"
    else:
        agg = f"sum(c.{spec['value']})::BIGINT AS agg_value"
    value_alias = "agg_value"

    con.execute(f"""
        CREATE TABLE pb AS
        SELECT {group_cols}, {agg}
        FROM read_parquet('{COUNT_PARQUET}') c
        JOIN om o ON c.obs_id = o.obs_id
        JOIN gene_map g ON c.gene = g.gene
        GROUP BY 1{', 2' if spec['half'] else ''}{', 3' if spec['half'] else ''}
    """)
    con.register("keys", keys)
    select_cols = ", ".join(f"k.{k}" for k in spec["keys"])
    out = os.path.join(OUT, spec["out"])
    con.execute(f"""
        COPY (
          SELECT {select_cols}{extra}, g.gene, p.{value_alias} AS
                 {'counts' if spec['value'] == 'count' else 'norm_sum'}
          FROM pb p
          JOIN keys k USING(group_id)
          JOIN gene_map g USING(gene_id)
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    print(f"[{variant}] groups={len(keys)} -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all",
                        choices=list(VARIANTS) + ["all"])
    args = parser.parse_args()
    ensure_out()
    obs = pd.read_csv(OBS_CSV)
    genes = pd.read_csv(os.path.join(OUT, "genes.csv"))
    gene_map = genes.copy()
    gene_map["gene_id"] = np.arange(1, len(gene_map) + 1, dtype=np.int32)
    for variant in (list(VARIANTS) if args.variant == "all" else [args.variant]):
        build(variant, obs, gene_map)


if __name__ == "__main__":
    main()
