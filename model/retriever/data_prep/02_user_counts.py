"""02, pass 1: per-user positive/hard-neg counts, streamed over ratings.csv.

Labels come from prep_config (the only source):
  positive = status in {completed, watching} and score not in [1,4]
  hard-neg = dropped, or score in [1,4] with any status
Writes _user_stats.parquet for the split (03). ratings.csv is large: polars lazy +
streaming, never a full collect.
"""
import pathlib

import polars as pl

from prep_config import EVAL_MIN_POS, is_hardneg_expr, is_pos_expr

RETRIEVER = pathlib.Path(__file__).parent.parent          # model/retriever/
ROOT = RETRIEVER.parent.parent                            # repo root
SRC = ROOT / "data" / "clean" / "ratings.csv"
OUT = RETRIEVER.parent.parent / "train" / "retriever" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    stats = (
        pl.scan_csv(SRC)
        .group_by("username")
        .agg(
            is_pos_expr().sum().alias("n_pos"),
            is_hardneg_expr().sum().alias("n_hardneg"),
        )
        .collect(engine="streaming")
        .with_columns(pl.col("n_pos").cast(pl.Int32), pl.col("n_hardneg").cast(pl.Int32))
    )
    stats.write_parquet(OUT / "_user_stats.parquet")

    n = stats.height
    n_zero = stats.filter(pl.col("n_pos") == 0).height
    n_eval = stats.filter(pl.col("n_pos") >= EVAL_MIN_POS).height
    print(f"users total: {n:,}")
    print(f"  n_pos == 0  (dropped from train): {n_zero:>9,}  ({n_zero / n * 100:5.2f}%)")
    print(f"  n_pos >= 1  (keep):               {n - n_zero:>9,}")
    print(f"  n_pos >= {EVAL_MIN_POS} (eval-eligible):      {n_eval:>9,}  ({n_eval / n * 100:5.2f}%)")
    print(f"  total positives: {int(stats['n_pos'].sum()):,}")
    print(f"  total hard-neg:  {int(stats['n_hardneg'].sum()):,}")
    print("DONE 02.")


if __name__ == "__main__":
    main()
