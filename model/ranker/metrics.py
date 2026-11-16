"""Two-stage recall/ndcg on a pool, plus rank_norm/blend. Pure numpy (no torch).

Mirrors model/retriever/metrics.py::evaluate exactly (binary relevance, mean per user, IDCG
normalized by min(R_total, K)); the only difference is ranking inside the top-D pool instead of
the full catalog. For K <= D, recall@K / ndcg@K equal the full-ranking values (the full ranking's
top-K is inside the pool). R_total = all of the user's queries, including those outside the pool,
so the denominator is unchanged.

Pool format (parquet written by build_eval/build_train): rows sorted by contiguous qid, one group
per user, each group sorted by cosine desc (pool_rank ascending).
"""
from __future__ import annotations

import numpy as np


def group_offsets(qid: np.ndarray) -> np.ndarray:
    """Contiguous qid (0..G-1, ascending) -> offsets [G+1]. The assert catches a badly written pool."""
    change = np.r_[True, qid[1:] != qid[:-1]]
    starts = np.flatnonzero(change)
    assert (qid[starts] == np.arange(len(starts))).all(), "qid must be dense and contiguous 0..G-1"
    return np.r_[starts, len(qid)]


def rank_norm(x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Rank normalized to [0, 1] per group (larger value -> closer to 1). Ties are broken by
    position, which is enough for blend."""
    out = np.empty(len(x), dtype=np.float64)
    for s, e in zip(offsets[:-1], offsets[1:]):
        n = e - s
        r = np.empty(n, dtype=np.float64)
        r[np.argsort(x[s:e], kind="stable")] = np.arange(n)
        out[s:e] = r / max(n - 1, 1)
    return out


def blend(cos: np.ndarray, pred: np.ndarray, offsets: np.ndarray, alpha: float) -> np.ndarray:
    """score = (1 - alpha) * rank_norm(cos) + alpha * rank_norm(pred) per group. Serving applies
    exactly this formula."""
    if alpha == 0.0:
        return cos.astype(np.float64)
    return (1 - alpha) * rank_norm(cos, offsets) + alpha * rank_norm(pred, offsets)


def eval_pool(scores: np.ndarray, labels: np.ndarray, offsets: np.ndarray,
              r_total: np.ndarray, ks, pooled: bool = False,
              label_liked: np.ndarray = None, r_liked: np.ndarray = None) -> dict:
    """Rerank each group by scores desc, then measure recall@K/ndcg@K, mean per user (labels are
    binary, 1 = candidate in the queries). pooled=True adds hitrate@K pooled over every
    (user, query) pair (for the thin cold slice).

    With label_liked (per candidate, 1 = candidate in the liked queries) and r_liked (liked queries
    per user, including those outside the pool), it adds liked_recall@K / liked_ndcg@K (report
    only) on the same ranking, over users with r_liked > 0 (n_users_liked). liked = score >= 1 and
    score > the user's mean (built in build_eval).

    Returns {recall@K, ndcg@K, n_users[, hitrate@K, n_pairs][, liked_recall@K, liked_ndcg@K, n_users_liked]}."""
    G = len(offsets) - 1
    kmax = max(ks)
    discount = 1.0 / np.log2(np.arange(2, kmax + 2))
    idcg_cum = np.cumsum(discount)

    sums = {f"recall@{k}": 0.0 for k in ks}
    sums.update({f"ndcg@{k}": 0.0 for k in ks})
    pooled_hits = {k: 0.0 for k in ks}
    n = 0
    total_rel = 0

    liked_on = label_liked is not None and r_liked is not None
    sums_liked = {f"liked_recall@{k}": 0.0 for k in ks}
    sums_liked.update({f"liked_ndcg@{k}": 0.0 for k in ks})
    n_liked = 0
    for g in range(G):
        s, e = offsets[g], offsets[g + 1]
        R = int(r_total[g])
        if R == 0:
            continue
        order = np.argsort(-scores[s:e], kind="stable")[:kmax]
        hit = labels[s:e][order].astype(np.float64)
        for k in ks:
            h = hit[:k]
            n_hit = h.sum()
            sums[f"recall@{k}"] += n_hit / R
            dcg = (h * discount[: len(h)][:k]).sum()
            sums[f"ndcg@{k}"] += dcg / idcg_cum[min(R, k) - 1]
            if pooled:
                pooled_hits[k] += n_hit
        total_rel += R
        n += 1

        if liked_on:
            RL = int(r_liked[g])
            if RL:
                hit_l = label_liked[s:e][order].astype(np.float64)
                for k in ks:
                    hl = hit_l[:k]
                    sums_liked[f"liked_recall@{k}"] += hl.sum() / RL
                    sums_liked[f"liked_ndcg@{k}"] += (hl * discount[: len(hl)][:k]).sum() \
                        / idcg_cum[min(RL, k) - 1]
                n_liked += 1

    out = {m: (v / n if n else 0.0) for m, v in sums.items()}
    out["n_users"] = n
    if pooled:
        for k in ks:
            out[f"hitrate@{k}"] = pooled_hits[k] / total_rel if total_rel else 0.0
        out["n_pairs"] = total_rel
    if liked_on:
        out.update({m: (v / n_liked if n_liked else 0.0) for m, v in sums_liked.items()})
        out["n_users_liked"] = n_liked
    return out


def load_pool_arrays(pool_path, users_path, k: int, max_groups: int | None = None):
    """Read a pool parquet, keeping the top k of each group by pool_rank. Returns
    (df, cos f64, label i8, offsets, r_total, users). max_groups keeps only qid < max_groups (smoke)."""
    import polars as pl

    df = pl.read_parquet(pool_path).filter(pl.col("pool_rank") < k)
    users = pl.read_parquet(users_path).sort("qid")
    if max_groups is not None:
        df = df.filter(pl.col("qid") < max_groups)
        users = users.filter(pl.col("qid") < max_groups)
    offsets = group_offsets(df["qid"].to_numpy())
    assert len(offsets) - 1 == users.height
    return (df, df["cos_uv"].to_numpy().astype(np.float64),
            df["label"].to_numpy().astype(np.int8), offsets,
            users["r_total"].to_numpy(), users)


def sweep_best_alpha(cos, pred, labels, offsets, r_total, ks, alphas) -> tuple[float, dict, dict]:
    """Sweep the blend alpha; returns (best alpha > 0 by ndcg@10, its metrics, {alpha: metrics})."""
    all_m = {a: eval_pool(blend(cos, pred, offsets, a), labels, offsets, r_total, ks)
             for a in alphas}
    best = max((a for a in alphas if a > 0), key=lambda a: all_m[a]["ndcg@10"])
    return best, all_m[best], all_m


def fmt(m: dict, ks) -> str:
    cols = " ".join(f"r@{k}={m[f'recall@{k}']:.4f}" for k in ks)
    nd = " ".join(f"ndcg@{k}={m[f'ndcg@{k}']:.4f}" for k in ks if k in (10, 100))
    return f"{cols}  {nd}  (n_users={m['n_users']:,})"
