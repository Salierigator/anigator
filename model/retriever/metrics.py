"""Eval on cold-user splits.

- Seen-mask: mask = seen(user) - the queries being scored, where seen covers every status
  (eval_seen.parquet). Masking less than that pushes the numbers down artificially.
- Two slices: warm (examples val/test, for tuning, headline recall@200) and cold (examples
  {val,test}_cold, the cold items H; test_cold is the final exam). Cold eval must refresh
  item_cache with cold_mask (H encoded as OOV) before calling evaluate.
- Cold also reports pooled hit-rate@K (the slice is thin, so per-user numbers are noisy) and
  a mode that ranks only H.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from model.retriever import data as data_mod


def group_examples(user_idx: np.ndarray, anime_idx: np.ndarray) -> Dict[int, List[int]]:
    """user_idx -> list of anime_idx (query items)."""
    out: Dict[int, List[int]] = {}
    for u, a in zip(user_idx.tolist(), anime_idx.tolist()):
        out.setdefault(u, []).append(a)
    return out


def group_scores(user_idx: np.ndarray, score: np.ndarray) -> Dict[int, List[int]]:
    """user_idx -> list of query scores, parallel to group_examples on the same user_idx (liked metric)."""
    out: Dict[int, List[int]] = {}
    for u, s in zip(user_idx.tolist(), score.tolist()):
        out.setdefault(u, []).append(s)
    return out


def build_masks(seen: Dict[int, np.ndarray], queries: Dict[int, List[int]]) -> Dict[int, np.ndarray]:
    """mask_ids[u] = seen[u] - queries[u]: items to drop from the ranking while scoring these
    queries. Queries are a subset of seen by construction, which is why seen is never masked as is."""
    out: Dict[int, np.ndarray] = {}
    for u, q in queries.items():
        s = seen.get(u)
        if s is None:
            out[u] = np.empty(0, dtype=np.int64)
        else:
            out[u] = np.setdiff1d(s, np.asarray(q, dtype=s.dtype))
    return out


def load_eval_protocol(train_data: Path, split: str):
    """Load one split ('val'|'test') once -> (queries_warm, mask_warm, queries_cold, mask_cold),
    shared by the model and the baselines."""
    seen = data_mod.load_eval_seen(train_data)
    ds_w = data_mod.ExamplesDataset(train_data, split)
    q_warm = group_examples(ds_w.user_idx, ds_w.anime_idx)
    ds_c = data_mod.ExamplesDataset(train_data, f"{split}_cold")
    q_cold = group_examples(ds_c.user_idx, ds_c.anime_idx)
    return q_warm, build_masks(seen, q_warm), q_cold, build_masks(seen, q_cold)


def load_query_scores(train_data: Path, split: str) -> Dict[int, List[int]]:
    """user_idx -> query scores of one split (liked metric), parallel to that split's queries."""
    ds = data_mod.ExamplesDataset(train_data, split)
    return group_scores(ds.user_idx, ds.score)


def support_mean(users, uids) -> Dict[int, float]:
    """u_mean per uid = mean rated support score (score >= 1) over the full history. score = 0
    (completed, unrated) is excluded; a user with nothing rated gets NaN, so has no liked queries."""
    out: Dict[int, float] = {}
    for u in uids:
        s, e = int(users.hist_offs[u]), int(users.hist_offs[u + 1])
        sc = users.hist_scores[s:e]
        rated = sc[sc >= 1]
        out[u] = float(rated.mean()) if len(rated) else float("nan")
    return out


def _pad_mask_ids(mask_ids: Dict[int, np.ndarray], chunk: List[int]) -> np.ndarray:
    """Stack the mask ids of a user chunk into a 0-padded matrix (PAD = 0 is never a candidate)."""
    mlen = max(max((len(mask_ids[u]) for u in chunk), default=0), 1)
    out = np.zeros((len(chunk), mlen), dtype=np.int64)
    for r, u in enumerate(chunk):
        m = mask_ids[u]
        out[r, : len(m)] = m
    return out


@torch.no_grad()
def evaluate(model, users, queries: Dict[int, List[int]], logq, ks, mask_ids: Dict[int, np.ndarray],
             eval_history_cap: int = 1024, batch: int = 512,
             candidate_mask: torch.Tensor = None, pooled: bool = False,
             query_scores: Dict[int, List[int]] = None):
    """model.item_cache must be refreshed in the right mode before the call
    (warm: refresh_item_cache(); cold: refresh_item_cache(cold_mask=...)).

    candidate_mask overrides the ranked set (None = isfinite(logq) = every real item; pass the
    cold_mask bool[N] to rank only H). pooled=True adds hitrate@K pooled over all (user, query)
    pairs plus n_pairs (for the thin cold slice).

    query_scores (score of each query item, parallel to queries) adds liked-recall@K and
    liked-ndcg@K, report only: a liked query has score >= 1 and score > u_mean (above the
    user's own mean over rated support). Ranking and masks are unchanged, so the numbers compare
    directly with the binary ones; only users with >= 1 liked query count (n_users_liked).

    Returns {recall@K, ndcg@K, n_users[, hitrate@K, n_pairs][, liked_recall@K, liked_ndcg@K, n_users_liked]}.
    """
    model.eval()
    device = model.item_cache.device
    kmax = max(ks)
    cand = torch.isfinite(logq).to(device) if candidate_mask is None else candidate_mask.to(device)

    uids = list(queries.keys())
    sums = {f"recall@{k}": 0.0 for k in ks}
    sums.update({f"ndcg@{k}": 0.0 for k in ks})
    pooled_hits = {k: 0.0 for k in ks}
    total_rel = 0
    n = 0

    liked_on = query_scores is not None
    u_mean = support_mean(users, uids) if liked_on else {}
    sums_liked = {f"liked_recall@{k}": 0.0 for k in ks}
    sums_liked.update({f"liked_ndcg@{k}": 0.0 for k in ks})
    n_liked = 0

    # IDCG[k] with every relevant item ranked first: sum_{i=1..k} 1/log2(i+1)
    discount = 1.0 / np.log2(np.arange(2, kmax + 2))
    idcg_cum = np.cumsum(discount)

    for s in range(0, len(uids), batch):
        chunk = uids[s : s + batch]
        u = np.asarray(chunk, dtype=np.int64)
        ids, hmask, hsc = users.eval_history_batch(u, eval_history_cap)
        ub = {
            "history_ids": torch.from_numpy(ids).to(device),
            "history_mask": torch.from_numpy(hmask).to(device),
            "history_scores": torch.from_numpy(hsc).to(device),
            "gender_id": torch.from_numpy(users.gender_id[u]).to(device),
            "joined_bucket": torch.from_numpy(users.joined_bucket[u]).to(device),
        }
        U = model.encode_users(ub)                              # [E, d]
        scores = U @ model.item_cache.t()                       # [E, N]
        scores[:, ~cand] = float("-inf")
        # mask seen - queries (never the queries: they are the answers being scored)
        mpad = torch.from_numpy(_pad_mask_ids(mask_ids, chunk)).to(device)
        scores.scatter_(1, mpad, float("-inf"))                 # pad 0 = PAD, never a candidate
        topk = torch.topk(scores, kmax, dim=1).indices.cpu().numpy()  # [E, kmax]

        for row, uid in enumerate(chunk):
            rel = set(queries[uid])
            R = len(rel)
            if R == 0:
                continue
            ranked = topk[row]
            hit = np.array([a in rel for a in ranked], dtype=np.float64)  # [kmax]
            for k in ks:
                h = hit[:k]
                n_hit = h.sum()
                sums[f"recall@{k}"] += n_hit / R
                dcg = (h * discount[:k]).sum()
                idcg = idcg_cum[min(R, k) - 1]
                sums[f"ndcg@{k}"] += dcg / idcg
                if pooled:
                    pooled_hits[k] += n_hit
            total_rel += R
            n += 1

            if liked_on:
                mu = u_mean[uid]
                liked = set() if np.isnan(mu) else {
                    a for a, sc in zip(queries[uid], query_scores[uid]) if sc >= 1 and sc > mu}
                R_liked = len(liked)
                if R_liked:
                    hit_l = np.array([a in liked for a in ranked], dtype=np.float64)
                    for k in ks:
                        hl = hit_l[:k]
                        sums_liked[f"liked_recall@{k}"] += hl.sum() / R_liked
                        sums_liked[f"liked_ndcg@{k}"] += (hl * discount[:k]).sum() \
                            / idcg_cum[min(R_liked, k) - 1]
                    n_liked += 1

    model.train()
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


def run_cold_eval(model, users, train_data: Path, logq, ks, split: str = "test",
                  eval_history_cap: int = 1024, h_only: bool = False):
    """Cold slice (final exam on test, val for debugging): refresh the cache in cold mode, then
    evaluate the cold queries (pooled). h_only=True ranks only H (content diagnostic).
    item_cache is left in cold mode: call refresh_item_cache() again before any warm use."""
    spec = data_mod.load_feature_spec(train_data)
    cold_mask = data_mod.load_cold_mask(train_data, spec["num_items"])
    _, _, q_cold, m_cold = load_eval_protocol(train_data, split)
    qs_cold = load_query_scores(train_data, f"{split}_cold")   # liked metric (diagnostic only on cold)
    model.refresh_item_cache(cold_mask=cold_mask)
    cand = cold_mask if h_only else None
    return evaluate(model, users, q_cold, logq, ks, m_cold,
                    eval_history_cap=eval_history_cap, candidate_mask=cand, pooled=True,
                    query_scores=qs_cold)
