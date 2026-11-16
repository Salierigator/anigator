"""03: cold-user split (90/5/5) + user_idx re-index.

Whole users are held out. Eval (val/test) takes only users with n_pos >= 11, enough for the
support/query split of 05; users with 1..10 positives always go to train, users with none
are dropped. The split is a stable hash of (username, SEED): reproducible, no RNG.

Reads _user_stats.parquet; writes _user_split.parquet (username, user_idx, split, n_pos,
n_hardneg) and user_id_map.parquet (username, user_idx), all in train/twotower/retriever/data/.
"""
import pathlib

import polars as pl

from prep_config import EVAL_MIN_POS, SEED

RETRIEVER = pathlib.Path(__file__).parent.parent          # model/twotower/retriever/
OUT = RETRIEVER.parents[2] / "train" / "twotower" / "retriever" / "data"


def main():
    stats = pl.read_parquet(OUT / "_user_stats.parquet")
    kept = stats.filter(pl.col("n_pos") >= 1)

    bucket = pl.col("username").hash(seed=SEED) % 100
    eligible = pl.col("n_pos") >= EVAL_MIN_POS
    split = (
        pl.when(~eligible).then(pl.lit("train"))      # 1..10 pos -> train
        .when(bucket < 90).then(pl.lit("train"))
        .when(bucket < 95).then(pl.lit("val"))
        .otherwise(pl.lit("test"))
    )

    kept = (
        kept.with_columns(split.alias("split"))
        .sort("username")
        .with_row_index("user_idx")
        .with_columns(pl.col("user_idx").cast(pl.Int32))
    )
    kept.write_parquet(OUT / "_user_split.parquet")
    kept.select("username", "user_idx").write_parquet(OUT / "user_id_map.parquet")

    n = kept.height
    print(f"kept users (n_pos>=1): {n:,}")
    for s in ["train", "val", "test"]:
        c = kept.filter(pl.col("split") == s).height
        print(f"  {s:<6} {c:>9,}  ({c / n * 100:5.2f}%)")
    # every eval user has >= EVAL_MIN_POS positives
    bad = kept.filter((pl.col("split") != "train") & (pl.col("n_pos") < EVAL_MIN_POS)).height
    assert bad == 0, f"{bad} eval users have n_pos < {EVAL_MIN_POS}"
    print(f"  eval-eligibility OK (every val/test user has >= {EVAL_MIN_POS} positives)")
    print("DONE 03.")


if __name__ == "__main__":
    main()
