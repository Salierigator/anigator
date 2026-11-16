"""Build the user vector U from export/ (shared by the ranker and the service).

Replicates the retriever's user-side encoder without touching its training data:
- histories come from export/users_history.parquet (full, sorted by score desc with ties
  ascending, so a prefix is top-by-score; for an eval user it is the support, queries and cold H
  removed). This module receives the history ids/scores ready-made.
- gender/joined use the maps stored in user_tower.pt (same as
  model/retriever/data_prep/04_user_features.py); encode_gender_joined is only needed for new
  users at serve time (known users already have them in users_history).
- U = UserTower(pool_history(history over item_vectors), gender, joined), L2-normalized.

Reads only export/ and imports only model definitions (UserTower), no training code.
pool_history/_attn_pool copy TwoTower.pool_history from model/retriever/model.py (TwoTower itself
needs the training ItemTable) and must stay in sync with it and with check_export.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.paths import EXPORT
from model.retriever.model import UserTower, _masked_mean, _masked_weighted_mean   # model definitions only


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

    def _attn_pool(self, vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = (self.attn_key(vecs) @ self.attn_query) / self.d ** 0.5
        logits = logits.masked_fill(~mask, float("-inf"))
        logits = logits.masked_fill(~mask.any(dim=-1, keepdim=True), 0.0)
        attn = torch.softmax(logits, dim=-1)
        return (attn.unsqueeze(-1) * vecs).sum(dim=-2)

    def pool_history(self, history_ids, history_mask, history_scores=None) -> torch.Tensor:
        vecs = (self.hist_emb(history_ids) if self.history_source == "embed"
                else self.item_cache[history_ids])             # [B, L, d]
        if self.history_pool == "attn":
            pooled = self._attn_pool(vecs, history_mask)
        elif self.score_pool == "none" or history_scores is None:
            pooled = _masked_mean(vecs, history_mask)
        else:
            if self.score_pool == "learned":
                w = F.softplus(self.score_weight(history_scores).squeeze(-1))
            else:
                w = history_scores.to(vecs.dtype).clamp(min=1.0)
            pooled = _masked_weighted_mean(vecs, history_mask, w)
        empty = ~history_mask.any(dim=-1)
        return torch.where(empty.unsqueeze(-1), self.user_tower.h_empty.to(pooled.dtype), pooled)

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
    model/retriever/data_prep/04_user_features.py.

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
