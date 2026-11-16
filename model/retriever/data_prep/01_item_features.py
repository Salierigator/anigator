"""01: item features, the anime id map and the cold-item set H for the Two-Tower retriever.

Reads data/clean/details.csv (pandas, the file is small), encodes every item feature and
re-indexes mal_id -> anime_idx (0 = PAD, 1 = OOV, real 2..N+1 sorted by mal_id). Also picks
the cold set H = the ~COLD_FRAC newest anime by start_date (a null date excludes it): 05 keeps
H out of training and eval uses it as the cold slice.
Writes anime_id_map.parquet, item_features.parquet, cold_items.parquet and _spec_item.json
(merged by 06) to train/retriever/data/.

Encoding is deterministic and the vocab maps are stored in the spec, so serving encodes
exactly the same way.
"""
import ast
import json
import math
import pathlib
from collections import Counter

import numpy as np
import pandas as pd
import polars as pl

from prep_config import COLD_FRAC

RETRIEVER = pathlib.Path(__file__).parent.parent          # model/retriever/
ROOT = RETRIEVER.parent.parent                            # repo root
SRC = ROOT / "data" / "clean" / "details.csv"
OUT = RETRIEVER.parent.parent / "train" / "retriever" / "data"
OUT.mkdir(parents=True, exist_ok=True)

PAD, OOV, FIRST_REAL = 0, 1, 2

# --- bucket edges ---
ERA_LABELS = ["<=1989", "1990-99", "2000-09", "2010-17", "2018+"]
ERA_BINS = [-np.inf, 1989, 1999, 2009, 2017, np.inf]
EP_LABELS = ["1", "2", "3-6", "7-13", "14-26", "27-52", "53+"]
EP_BINS = [0, 1, 2, 6, 13, 26, 52, np.inf]

STUDIO_MIN_COUNT = 10  # studios seen >= 10 times keep their own id (300 of them); the rest share OOV

# embedding dims; the multi-hot genres/themes have no dim: they are fed as raw vectors.
DIMS = {"type": 4, "source": 8, "rating": 4, "demographics": 4,
        "start_year": 4, "episodes": 4, "studios": 16}


def parse_list(v):
    """'[]'/NaN -> []; "['A','B']" -> ['A','B']."""
    if pd.isna(v):
        return []
    try:
        out = ast.literal_eval(v)
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def encode_cat(series):
    """Distinct non-null values -> 1..k (sorted); null/unseen -> 0 (OOV)."""
    vals = sorted(map(str, series.dropna().unique()))
    vmap = {v: i + 1 for i, v in enumerate(vals)}  # 0 = OOV
    codes = series.map(lambda v: vmap.get(str(v), 0) if pd.notna(v) else 0)
    return codes.to_numpy(np.int16), vmap


def encode_demographics(series):
    """Single tag (the first one). empty -> 0 ('none'); the 5 tags -> 1..5. No OOV (closed set)."""
    first = series.map(parse_list).map(lambda L: L[0] if L else None)
    tags = sorted(t for t in first.dropna().unique())
    tmap = {t: i + 1 for i, t in enumerate(tags)}  # 0 = none/empty
    codes = first.map(lambda t: tmap.get(t, 0) if t is not None else 0)
    return codes.to_numpy(np.int16), tmap


def encode_multihot(series):
    """Multi-hot over the sorted tag set, plus a final 'present' column."""
    parsed = series.map(parse_list)
    tags = sorted({t for L in parsed for t in L})
    tmap = {t: i for i, t in enumerate(tags)}
    present_idx = len(tags)
    width = len(tags) + 1
    mat = np.zeros((len(series), width), dtype=np.int8)
    for r, L in enumerate(parsed):
        for t in L:
            mat[r, tmap[t]] = 1
        if L:
            mat[r, present_idx] = 1
    return mat, tmap, present_idx, width


def encode_studios(series):
    """Multi-value (average-pooled in the model): 0 = empty, 1 = OOV, 2.. = studios seen >= 10
    times; one list of ids per anime."""
    parsed = series.map(parse_list)
    cnt = Counter(t for L in parsed for t in L)
    kept = sorted(t for t, c in cnt.items() if c >= STUDIO_MIN_COUNT)
    smap = {t: i + 2 for i, t in enumerate(kept)}  # 0=empty, 1=OOV
    ids = parsed.map(
        lambda L: [0] if not L else sorted({smap.get(t, 1) for t in L})
    )
    return ids.tolist(), smap


def bucketize(series, bins, null_id=0):
    """pd.cut -> codes 0..(nbins-1), shifted by 1 to keep id 0 for NULL; NaN -> null_id."""
    code = pd.cut(series, bins=bins, labels=False, right=True, include_lowest=True)
    return (code + 1).fillna(null_id).to_numpy(np.int16)


def main():
    df = pd.read_csv(SRC).sort_values("mal_id").reset_index(drop=True)
    n = len(df)
    print(f"details rows: {n:,}")

    anime_idx = np.arange(FIRST_REAL, FIRST_REAL + n, dtype=np.int32)

    type_codes, type_map = encode_cat(df["type"])
    source_codes, source_map = encode_cat(df["source"])
    rating_codes, rating_map = encode_cat(df["rating"])
    demo_codes, demo_map = encode_demographics(df["demographics"])

    dates = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    year = dates.dt.year
    startyear_bucket = bucketize(year, ERA_BINS)
    episodes_bucket = bucketize(df["episodes"], EP_BINS)

    genres_mat, genre_map, genre_present, genre_w = encode_multihot(df["genres"])
    themes_mat, theme_map, theme_present, theme_w = encode_multihot(df["themes"])
    studio_ids, studio_map = encode_studios(df["studios"])

    # --- vocab sizes must match the expected ones (catches data drift early) ---
    checks = {
        "type": (len(type_map) + 1, 10), "source": (len(source_map) + 1, 18),
        "rating": (len(rating_map) + 1, 7), "demographics": (len(demo_map) + 1, 6),
        "start_year": (len(ERA_LABELS) + 1, 6), "episodes": (len(EP_LABELS) + 1, 8),
        "genres_width": (genre_w, 22), "themes_width": (theme_w, 53),
        "studios": (len(studio_map) + 2, 302),
    }
    print("\nvocab/width check (got vs expected):")
    for name, (got, exp) in checks.items():
        flag = "OK" if got == exp else "!! MISMATCH"
        print(f"  {name:<14} {got:>4}  (expected {exp})  {flag}")
        assert got == exp, f"{name}: got {got}, expected {exp}"

    # --- 2 special rows (PAD, OOV): every feature neutral = 0 ---
    sp_scalar = {k: [0, 0] for k in
                 ["type_id", "source_id", "rating_id", "demographics_id",
                  "startyear_bucket", "episodes_bucket"]}
    table = {
        "anime_idx": [PAD, OOV] + anime_idx.tolist(),
        "type_id": sp_scalar["type_id"] + type_codes.tolist(),
        "source_id": sp_scalar["source_id"] + source_codes.tolist(),
        "rating_id": sp_scalar["rating_id"] + rating_codes.tolist(),
        "demographics_id": sp_scalar["demographics_id"] + demo_codes.tolist(),
        "startyear_bucket": sp_scalar["startyear_bucket"] + startyear_bucket.tolist(),
        "episodes_bucket": sp_scalar["episodes_bucket"] + episodes_bucket.tolist(),
        "genres_multihot": [[0] * genre_w, [0] * genre_w] + [r.tolist() for r in genres_mat],
        "themes_multihot": [[0] * theme_w, [0] * theme_w] + [r.tolist() for r in themes_mat],
        "studio_ids": [[0], [0]] + studio_ids,
    }
    items = pl.DataFrame(table).with_columns(
        pl.col("anime_idx").cast(pl.Int32),
        pl.col("type_id").cast(pl.Int8), pl.col("source_id").cast(pl.Int8),
        pl.col("rating_id").cast(pl.Int8), pl.col("demographics_id").cast(pl.Int8),
        pl.col("startyear_bucket").cast(pl.Int8), pl.col("episodes_bucket").cast(pl.Int8),
        pl.col("genres_multihot").cast(pl.List(pl.Int8)),
        pl.col("themes_multihot").cast(pl.List(pl.Int8)),
        pl.col("studio_ids").cast(pl.List(pl.Int32)),
    )
    items.write_parquet(OUT / "item_features.parquet")
    print(f"\nitem_features.parquet: {items.height:,} rows (= {n} real + 2 PAD/OOV)")
    assert items.height == n + 2

    pl.DataFrame({
        "mal_id": df["mal_id"].to_numpy(),
        "anime_idx": anime_idx,
    }).with_columns(pl.col("mal_id").cast(pl.Int64), pl.col("anime_idx").cast(pl.Int32)
                    ).write_parquet(OUT / "anime_id_map.parquet")
    print(f"anime_id_map.parquet: {n:,} rows")

    # --- cold-item set H: the COLD_FRAC newest anime by start_date (null dates are not candidates).
    # Ties on one date break by df order (mal_id asc, stable mergesort), so H is deterministic.
    n_cold = math.ceil(COLD_FRAC * n)
    valid = dates.notna()
    newest = dates[valid].sort_values(ascending=False, kind="mergesort")
    cold_rows = newest.index[:n_cold]
    cold_mask = np.zeros(n, dtype=bool)
    cold_mask[cold_rows] = True
    cutoff = newest.iloc[n_cold - 1]
    pl.DataFrame({
        "anime_idx": anime_idx[cold_mask],
        "mal_id": df.loc[cold_mask, "mal_id"].to_numpy(),
        "start_date": df.loc[cold_mask, "start_date"].astype(str).to_numpy(),
    }).with_columns(
        pl.col("anime_idx").cast(pl.Int32), pl.col("mal_id").cast(pl.Int64),
    ).write_parquet(OUT / "cold_items.parquet")
    print(f"cold_items.parquet: {n_cold:,} anime (frac {COLD_FRAC}, cutoff {cutoff.date()}, "
          f"{int((~valid).sum())} null-date excluded)")

    spec_item = {
        "special_idx": {"PAD": PAD, "OOV": OOV, "first_real": FIRST_REAL},
        "num_items": n + 2,
        "cold_items": {
            "frac": COLD_FRAC,
            "n_cold": int(n_cold),
            "cutoff_date": str(cutoff.date()),
            "n_null_date_excluded": int((~valid).sum()),
        },
        "item_features": {
            "type": {"kind": "cat", "vocab": len(type_map) + 1, "dim": DIMS["type"],
                     "oov_id": 0, "map": type_map},
            "source": {"kind": "cat", "vocab": len(source_map) + 1, "dim": DIMS["source"],
                       "oov_id": 0, "map": source_map},
            "rating": {"kind": "cat", "vocab": len(rating_map) + 1, "dim": DIMS["rating"],
                       "oov_id": 0, "map": rating_map},
            "demographics": {"kind": "cat", "vocab": len(demo_map) + 1, "dim": DIMS["demographics"],
                             "none_id": 0, "map": demo_map},
            "start_year": {"kind": "bucket", "vocab": len(ERA_LABELS) + 1, "dim": DIMS["start_year"],
                           "null_id": 0, "bins": [str(b) for b in ERA_BINS], "labels": ERA_LABELS},
            "episodes": {"kind": "bucket", "vocab": len(EP_LABELS) + 1, "dim": DIMS["episodes"],
                         "null_id": 0, "bins": [str(b) for b in EP_BINS], "labels": EP_LABELS},
            "genres": {"kind": "multihot", "width": genre_w, "present_index": genre_present,
                       "map": genre_map},
            "themes": {"kind": "multihot", "width": theme_w, "present_index": theme_present,
                       "map": theme_map},
            "studios": {"kind": "multivalue", "vocab": len(studio_map) + 2, "dim": DIMS["studios"],
                        "empty_id": 0, "oov_id": 1, "map": studio_map},
        },
    }
    (OUT / "_spec_item.json").write_text(json.dumps(spec_item, ensure_ascii=False, indent=2))
    print(f"_spec_item.json written ({len(studio_map)} studios kept, occ>={STUDIO_MIN_COUNT})")
    print("DONE 01.")


if __name__ == "__main__":
    main()
