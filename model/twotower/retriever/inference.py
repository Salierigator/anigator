"""The retriever as loaded from the bundle: encode a user, take the cosine top-K.

This is the only module the ranker and the serving code import from the retriever. It reads only
export/twotower/ (user_tower.pt + item_vectors.npy) and imports only model definitions (UserTower),
no training code.

- Histories arrive ready-made as ids/scores: full lists sorted by score desc with ties ascending,
  so a prefix is top-by-score.
- gender/joined use the maps stored in user_tower.pt (same as
  model/twotower/retriever/data_prep/04_user_features.py); encode_gender_joined is only needed for
  new users at serve time.
- U = UserTower(pool_history(history over item_vectors), gender, joined), L2-normalized.

History pooling is core/model.py::pool_history, the function training uses (TwoTower itself needs
the training ItemTable, so it cannot be loaded from the bundle).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model.twotower.paths import EXPORT
from model.twotower.retriever.core.model import UserTower, pool_history   # model definitions only


class UserEncoder(nn.Module):
    """Mirrors the user branch of TwoTower: pooled history (mean/attn, optionally score-weighted)
    + gender + joined -> U[d]."""

    def __init__(self, meta: dict, item_vectors: np.ndarray):
        super().__init__()
        self.d = meta["d"]
        self.history_pool = meta["history_pool"]
        self.score_pool = meta["score_pool"]
        self.history_source = meta.get("history_source", "cache")
        self.user_tower = UserTower({"user_features": meta["user_features"]}, meta["d"], meta["mlp_hidden"])
        if self.history_pool == "attn":
            self.attn_key = nn.Linear(self.d, self.d, bias=False)
            self.attn_query = nn.Parameter(torch.zeros(self.d))
        if self.score_pool == "learned":
            self.score_weight = nn.Embedding(11, 1)
        if self.history_source == "embed":
            # Trainable history table (exported as hist_emb.*), distinct from item_cache.
            self.hist_emb = nn.Embedding(item_vectors.shape[0], self.d, padding_idx=0)
        # item_cache = item_vectors (row == anime_idx); non-persistent so load_state_dict ignores it.
        # Kept even with history_source="embed": it is also the catalog-wide scoring table.
        self.register_buffer("item_cache", torch.from_numpy(item_vectors).float(), persistent=False)

    def pool_history(self, history_ids, history_mask, history_scores=None) -> torch.Tensor:
        vecs = (self.hist_emb(history_ids) if self.history_source == "embed"
                else self.item_cache[history_ids])             # [B, L, d]
        return pool_history(
            vecs, history_mask, history_scores, history_pool=self.history_pool,
            score_pool=self.score_pool, d=self.d, attn_key=getattr(self, "attn_key", None),
            attn_query=getattr(self, "attn_query", None),
            score_weight=getattr(self, "score_weight", None), h_empty=self.user_tower.h_empty)

    @torch.no_grad()
    def encode(self, history_ids, history_mask, history_scores, gender_id, joined_bucket) -> torch.Tensor:
        pooled = self.pool_history(history_ids, history_mask, history_scores)
        return self.user_tower(pooled, gender_id, joined_bucket)


def load_user_encoder(device: str = "cpu", artifacts: Path = EXPORT):
    """Load user_tower.pt + item_vectors.npy -> (UserEncoder in eval mode, meta)."""
    meta = torch.load(artifacts / "user_tower.pt", map_location="cpu", weights_only=False)
    item_vectors = np.load(artifacts / "item_vectors.npy")
    enc = UserEncoder(meta, item_vectors)
    enc.load_state_dict(meta["state_dict"], strict=True)
    return enc.eval().to(device), meta


def encode_gender_joined(prof: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """prof[username, gender, joined] -> [username, gender_id, joined_bucket], as in
    model/twotower/retriever/data_prep/04_user_features.py.

    gender: map from meta (unknown -> 0 = OOV). joined: year -> bucket by the cohort bins in
    meta; null/unparseable -> the newest cohort (id n_bucket-1)."""
    gmap = meta["user_features"]["gender"]["map"]
    bins = [float(b) for b in meta["user_features"]["joined"]["bins"]]   # [-inf,2012,...,inf]
    newest = len(bins) - 2                                               # id of the newest cohort (n_bucket-1)

    out = prof.copy()
    out["gender_id"] = out["gender"].map(
        lambda v: gmap.get(v, 0) if pd.notna(v) else 0).astype("int64")
    jyear = pd.to_datetime(out["joined"], errors="coerce").dt.year
    jcode = pd.cut(jyear, bins=bins, labels=False, right=True)           # 0..n_bucket-1
    out["joined_bucket"] = jcode.fillna(newest).astype("int64")
    return out[["username", "gender_id", "joined_bucket"]]


def pad_lists(lists: list[np.ndarray], cap: int, dtype=np.int64):
    """Ragged lists -> (mat [n, W] zero-padded, mask [n, W]); W = min(max len, cap), at least 1."""
    W = max(min(max((len(x) for x in lists), default=1), cap), 1)
    mat = np.zeros((len(lists), W), dtype=dtype)
    mask = np.zeros((len(lists), W), dtype=bool)
    for i, x in enumerate(lists):
        v = np.asarray(x)[:W]
        mat[i, : len(v)] = v
        mask[i, : len(v)] = True
    return mat, mask


@torch.no_grad()
def encode_users(enc, hist_ids: list[np.ndarray], hist_scores: list[np.ndarray],
                 gender: np.ndarray, joined: np.ndarray, cap: int) -> torch.Tensor:
    """U [n, d] from the history prefix (sorted by score desc), as in the retriever's eval path."""
    ids, mask = pad_lists(hist_ids, cap)
    sc, _ = pad_lists(hist_scores, cap)
    return enc.encode(torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(sc),
                      torch.from_numpy(gender), torch.from_numpy(joined))


@torch.no_grad()
def topk_pool(U: torch.Tensor, item_cache: torch.Tensor, mask_lists: list[np.ndarray],
              depth: int, cold_idx: np.ndarray | None = None):
    """scores = U @ V.T with -inf for PAD/OOV, the per-user mask and every index in cold_idx
    (if given); returns the top-depth, sorted desc, as (cand [n, depth] i64, cos [n, depth] f32)."""
    scores = U @ item_cache.t()                                   # [n, N]
    scores[:, :2] = float("-inf")
    if cold_idx is not None:
        scores[:, torch.from_numpy(cold_idx)] = float("-inf")
    mpad, mmask = pad_lists(mask_lists, cap=10**9)
    mpad[~mmask] = 0                                              # padding points at PAD (already -inf)
    scores.scatter_(1, torch.from_numpy(mpad), float("-inf"))
    top = torch.topk(scores, depth, dim=1)                        # sorted desc
    return top.indices.numpy().astype(np.int64), top.values.numpy().astype(np.float32)
