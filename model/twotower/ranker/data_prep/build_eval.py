"""Build the eval pools (val/test/val_cold) with exactly the retriever's protocol. Re-run whenever
export/twotower/ changes; reads only export/twotower/.

Per eval user: U from the support (= its users_history list, queries and cold H removed) ->
top-POOL_DEPTH cosine with mask = seen - queries (queries are not masked: they are the answers
being scored; cold items are not removed: warm eval lets them compete exactly as in the
retriever) -> features. Pools are stored 500 deep; eval slices 200 or 500.

Writes train/twotower/ranker/data/pools/eval_{split}.parquet (contiguous qid, each group sorted by cosine
desc, binary label = candidate in the queries), eval_{split}_users.parquet (qid, user_idx,
r_total, r_liked, U, hist_top64) and build_meta.json.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import torch  # noqa: F401  (torch before lightgbm, as everywhere)

from model.twotower.ranker import config
from model.twotower.ranker.core.features import ItemFeatures, build_frame, cross_features, user_stats_from_support
from model.twotower.ranker.core.pools import PoolWriter
from model.twotower.ranker.data_prep.bundle import UsersHistory, account_age_by_user, load_eval_seen, load_queries
from model.twotower.retriever.inference import encode_users, load_user_encoder, topk_pool


def build_split(split: str, enc, cap: int, itemfeat, V, uh: UsersHistory,
                seen: dict, ages: dict) -> int:
    t0 = time.time()
    queries, query_scores = load_queries(split)
    uids = np.asarray(sorted(queries), dtype=np.int64)
    writer = PoolWriter(config.POOLS / f"eval_{split}.parquet",
                        config.POOLS / f"eval_{split}_users.parquet")
    for s in range(0, len(uids), config.CHUNK):
        chunk = uids[s : s + config.CHUNK]
        hist_ids = [uh.history(u)[0] for u in chunk]
        hist_scores = [uh.history(u)[1] for u in chunk]
        U = encode_users(enc, hist_ids, hist_scores,
                         uh.gender_id[chunk], uh.joined_bucket[chunk], cap)
        mask_lists = [np.setdiff1d(seen[u], queries[u]) for u in chunk]
        cand, cos = topk_pool(U, enc.item_cache, mask_lists, config.POOL_DEPTH)
        labels = np.stack([np.isin(c, queries[u]) for c, u in zip(cand, chunk)]).astype(np.int8)
        stats = user_stats_from_support(
            hist_scores, np.asarray([ages.get(int(u), np.nan) for u in chunk]))
        # liked = query items with score >= 1 and score > the user's mean (as in the retriever);
        # u_n_rated == 0 (no rated support) makes the mean meaningless, so nothing is liked.
        liked_sets = []
        for i, u in enumerate(chunk):
            if stats["u_n_rated"][i] > 0:
                qa, qsc = queries[u], query_scores[u]
                liked_sets.append(qa[(qsc >= 1) & (qsc > stats["u_mean_score"][i])])
            else:
                liked_sets.append(np.empty(0, dtype=np.int64))
        label_liked = np.stack([np.isin(c, ls) for c, ls in zip(cand, liked_sets)]).astype(np.int8)
        r_liked = np.asarray([len(ls) for ls in liked_sets], dtype=np.int64)
        cross = cross_features(V, itemfeat, cand, cos, hist_ids, stats)
        frame = build_frame(itemfeat, cand.ravel(), cross)
        r_total = np.asarray([len(queries[u]) for u in chunk])
        writer.add_chunk(chunk, cand, labels, frame, U.numpy(), hist_ids, r_total,
                         label_liked=label_liked, r_liked=r_liked)
        print(f"  [{split}] {min(s + config.CHUNK, len(uids)):,}/{len(uids):,} users "
              f"({time.time() - t0:.0f}s)", flush=True)
    n = writer.close()
    print(f"[done] eval_{split}: {n:,} users, depth {config.POOL_DEPTH} "
          f"({time.time() - t0:.0f}s)")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Build eval pools (val/test/val_cold)")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "val_cold"])
    args = ap.parse_args()
    assert "test_cold" not in args.splits or \
        (config.EXPORT / "eval_queries_test_cold.parquet").exists(), \
        "test_cold is the final exam: it needs the retriever's export --final-exam (scored once, when the pipeline is final)"

    enc, meta = load_user_encoder("cpu")
    cap = meta.get("eval_history_cap", 1024)
    V = enc.item_cache.numpy()
    itemfeat = ItemFeatures.load(config.EXPORT)
    uh = UsersHistory()
    seen = load_eval_seen()
    ages = account_age_by_user()

    counts = {s: build_split(s, enc, cap, itemfeat, V, uh, seen, ages) for s in args.splits}

    meta_path = config.POOLS / "build_meta.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    contract = (config.EXPORT / "CONTRACT.md").read_text().splitlines()
    old.update({
        "pool_depth": config.POOL_DEPTH, "hist_feat_cap": config.HIST_FEAT_CAP,
        "n_users": {**old.get("n_users", {}), **counts},
        "source_checkpoint": next((l.strip("- ") for l in contract if "Source checkpoint" in l), "?"),
        "artifacts_generated": next((l.strip("- ") for l in contract if "Generated" in l), "?"),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    meta_path.write_text(json.dumps(old, indent=2))
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
