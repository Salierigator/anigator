"""99: verify every prepared artifact in train/twotower/retriever/data/ (reads only that folder).

Checks schema/dtypes, ids within [0, vocab), multi-hot widths, examples with anime_idx >= 2 and
valid user_idx, aligned and score-sorted histories, logq; the cold items (valid ids; warm
train/val/test examples, histories and hard_neg all disjoint from H; cold examples within H
and from users of the right split); eval_seen covering history + warm and cold queries.
Raises on the first failure.
"""
import json
import pathlib

import polars as pl

RETRIEVER = pathlib.Path(__file__).parent.parent          # model/twotower/retriever/
OUT = RETRIEVER.parents[2] / "train" / "twotower" / "retriever" / "data"


def main():
    spec = json.loads((OUT / "feature_spec.json").read_text())
    num_items, num_users = spec["num_items"], spec["num_users"]
    ok = []

    # ---- item_features ----
    items = pl.read_parquet(OUT / "item_features.parquet")
    assert items.height == num_items, f"item rows {items.height} != {num_items}"
    scalar = {"type": "type_id", "source": "source_id", "rating": "rating_id",
              "demographics": "demographics_id", "start_year": "startyear_bucket",
              "episodes": "episodes_bucket"}
    for feat, col in scalar.items():
        vocab = spec["item_features"][feat]["vocab"]
        lo, hi = items[col].min(), items[col].max()
        assert lo >= 0 and hi < vocab, f"{col}: [{lo},{hi}] outside [0,{vocab})"
    for feat, col, w in [("genres", "genres_multihot", 22), ("themes", "themes_multihot", 53)]:
        assert spec["item_features"][feat]["width"] == w
        lens = items.select(pl.col(col).list.len().alias("l"))["l"]
        assert lens.min() == w and lens.max() == w, f"{col} width != {w}"
        mx = items.select(pl.col(col).list.max().alias("m"))["m"].max()
        assert mx <= 1, f"{col} is not 0/1 (max {mx})"
    svocab = spec["item_features"]["studios"]["vocab"]
    smax = items.select(pl.col("studio_ids").list.max().alias("m"))["m"].max()
    smin = items.select(pl.col("studio_ids").list.min().alias("m"))["m"].min()
    assert 0 <= smin and smax < svocab, f"studio id outside [0,{svocab})"
    ok.append(f"item_features: {items.height:,} rows, every id in [0,vocab), multihot 22/53 OK")

    # ---- cold_items ----
    cold_df = pl.read_parquet(OUT / "cold_items.parquet")
    cold_s = cold_df["anime_idx"]
    cold = cold_s.to_list()
    assert cold_df.height == spec["cold_items"]["n_cold"], "cold_items height != spec"
    assert cold_s.min() >= 2 and cold_s.max() < num_items, "cold anime_idx outside [2,num_items)"
    assert cold_s.n_unique() == cold_df.height, "cold_items has duplicates"
    ok.append(f"cold_items: {cold_df.height:,} items in [2,{num_items}), unique OK")

    # ---- users ----
    users = pl.read_parquet(OUT / "users.parquet")
    assert users.height == num_users
    chk = users.select(
        (pl.col("history_ids").list.len() == pl.col("history_scores").list.len()).all().alias("a"),
        pl.col("gender_id").max().alias("gmax"), pl.col("joined_bucket").max().alias("jmax"),
        pl.col("hard_neg_ids").list.len().max().alias("hncap"),
    )
    assert chk["a"].item(), "history_ids/scores lengths differ"
    assert chk["gmax"].item() < spec["user_features"]["gender"]["vocab"]
    assert chk["jmax"].item() < spec["user_features"]["joined"]["vocab"]
    assert chk["hncap"].item() <= spec["hard_neg_cap"]
    # sorted by score desc: consecutive diffs <= 0 (rows with len < 2 have a null max diff -> 0)
    mono = users.select(
        (pl.col("history_scores").list.eval(pl.element().cast(pl.Int16).diff())
         .list.max().fill_null(0) <= 0).all().alias("m")
    )["m"].item()
    assert mono, "history_scores not sorted desc"
    ok.append(f"users: {users.height:,} rows, history aligned + sorted desc, "
              f"gender/joined/hardneg in-range OK")

    # ---- H-isolation: history + hard_neg ----
    hist_exp = users.select("user_idx", "split", pl.col("history_ids").alias("aid")).explode("aid")
    hist_exp = hist_exp.drop_nulls("aid")
    assert hist_exp.filter(pl.col("aid").is_in(cold)).height == 0, "history contains a cold item"
    hn_exp = users.select(pl.col("hard_neg_ids").alias("aid")).explode("aid").drop_nulls("aid")
    assert hn_exp.filter(pl.col("aid").is_in(cold)).height == 0, "hard_neg contains a cold item"
    ok.append("H-isolation: history and hard_neg disjoint from H OK")

    # ---- examples warm ----
    total_ex = 0
    for name in ["train", "val", "test"]:
        ex = pl.read_parquet(OUT / "examples" / f"split={name}" / "part-0.parquet")
        assert ex.filter(pl.col("anime_idx") < 2).height == 0, f"{name}: PAD/OOV as a target"
        assert ex.filter((pl.col("user_idx") < 0) | (pl.col("user_idx") >= num_users)).height == 0
        assert ex.filter(pl.col("anime_idx").is_in(cold)).height == 0, f"{name}: contains a cold item"
        total_ex += ex.height
    ok.append(f"examples warm: {total_ex:,} rows, anime_idx>=2, disjoint from H OK")

    # ---- examples cold: within H, users of the right split ----
    user_split = users.select("user_idx", "split")
    for name, want in [("val_cold", "val"), ("test_cold", "test")]:
        ex = pl.read_parquet(OUT / "examples" / f"split={name}" / "part-0.parquet")
        assert ex.filter(~pl.col("anime_idx").is_in(cold)).height == 0, f"{name}: item outside H"
        bad = ex.join(user_split, on="user_idx").filter(pl.col("split") != want).height
        assert bad == 0, f"{name}: {bad} rows of users outside split {want}"
        ok.append(f"examples {name}: {ex.height:,} pairs within H, users in {want} OK")

    # ---- eval_seen covers history + warm queries + cold queries ----
    seen_exp = pl.read_parquet(OUT / "eval_seen.parquet").explode("seen_ids")
    n_eval = users.filter(pl.col("split") != "train").height
    assert seen_exp["user_idx"].n_unique() == n_eval, "eval_seen is missing users"
    need = pl.concat([
        hist_exp.filter(pl.col("split") != "train").select("user_idx", "aid"),
        *[
            pl.read_parquet(OUT / "examples" / f"split={s}" / "part-0.parquet")
            .select("user_idx", pl.col("anime_idx").alias("aid"))
            for s in ["val", "test", "val_cold", "test_cold"]
        ],
    ])
    missing = need.join(seen_exp, left_on=["user_idx", "aid"],
                        right_on=["user_idx", "seen_ids"], how="anti")
    assert missing.height == 0, f"eval_seen is missing {missing.height} pairs (history/query)"
    ok.append(f"eval_seen: {n_eval:,} users, covers history + queries OK")

    # ---- leak: no eval example in the history ----
    eval_ex = pl.concat([
        pl.read_parquet(OUT / "examples" / f"split={s}" / "part-0.parquet")
        for s in ["val", "test", "val_cold", "test_cold"]
    ])
    leak = eval_ex.join(hist_exp.select("user_idx", "aid"),
                        left_on=["user_idx", "anime_idx"],
                        right_on=["user_idx", "aid"], how="inner")
    assert leak.height == 0, "LEAK: an eval example is in the history"
    ok.append("leak (eval example in history) = 0 OK")

    # ---- logq ----
    logq = pl.read_parquet(OUT / "logq.parquet")
    assert logq.height == num_items
    assert logq.filter(~pl.col("is_candidate")).select((pl.col("anime_idx") < 2).all()).item()
    real = logq.filter(pl.col("anime_idx") >= 2)
    assert real.select(pl.col("log_q").is_finite().all()).item(), "a real item has a non-finite log_q"
    cold_cnt = logq.filter(pl.col("anime_idx").is_in(cold))["count"]
    assert cold_cnt.max() == 0, "a cold item has count > 0 in the train logq (leak)"
    s = float(logq.filter(pl.col("count") > 0).select(pl.col("log_q").exp().sum()).item())
    assert abs(s - 1.0) < 1e-6
    ok.append(f"logq: {logq.height:,} rows, real finite, cold count=0, sum(exp)=1.0 OK")

    print("=" * 64)
    for line in ok:
        print("  [OK] " + line)
    print("=" * 64)
    print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
