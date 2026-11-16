"""Readers for the bundle tables the ranker's data prep and eval need: user histories, eval seen
lists, eval queries and account ages. Reads only export/twotower/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from model.twotower.ranker import config
from model.twotower.ranker.core.features import REF_YEAR


class UsersHistory:
    """export/twotower/users_history.parquet, kept ragged (values + offsets, no full padding in RAM).
    Histories are full and sorted (score desc, ties ascending), so a prefix is top-by-score. For an
    eval user the list is the retriever's support (queries and cold H already removed)."""

    def __init__(self, artifacts: Path = config.EXPORT):
        t = pq.read_table(artifacts / "users_history.parquet")
        uidx = t.column("user_idx").to_numpy()
        self.num_users = len(uidx)
        assert (uidx == np.arange(self.num_users)).all(), "users_history must be dense 0..U-1"
        self.gender_id = t.column("gender_id").to_numpy().astype(np.int64)
        self.joined_bucket = t.column("joined_bucket").to_numpy().astype(np.int64)
        hcol = t.column("history_ids").combine_chunks()
        self.hist_vals = hcol.values.to_numpy(zero_copy_only=False).astype(np.int32)
        self.hist_offs = hcol.offsets.to_numpy().astype(np.int64)
        scol = t.column("history_scores").combine_chunks()
        self.hist_scores = scol.values.to_numpy(zero_copy_only=False).astype(np.int8)
        ncol = t.column("hard_neg_ids").combine_chunks()
        self.hn_vals = ncol.values.to_numpy(zero_copy_only=False).astype(np.int32)
        self.hn_offs = ncol.offsets.to_numpy().astype(np.int64)

    def history(self, u: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.hist_offs[u], self.hist_offs[u + 1]
        return self.hist_vals[s:e], self.hist_scores[s:e]

    def hard_neg(self, u: int) -> np.ndarray:
        return self.hn_vals[self.hn_offs[u]:self.hn_offs[u + 1]]

    def history_lens(self) -> np.ndarray:
        return self.hist_offs[1:] - self.hist_offs[:-1]


def load_eval_seen(artifacts: Path = config.EXPORT) -> dict[int, np.ndarray]:
    t = pq.read_table(artifacts / "eval_seen.parquet")
    uids = t.column("user_idx").to_numpy()
    col = t.column("seen_ids").combine_chunks()
    vals = col.values.to_numpy(zero_copy_only=False).astype(np.int64)
    offs = col.offsets.to_numpy().astype(np.int64)
    return {int(u): vals[offs[i]:offs[i + 1]] for i, u in enumerate(uids)}


def load_queries(split: str, artifacts: Path = config.EXPORT
                 ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """eval_queries_{split}.parquet -> (queries, query_scores): {user_idx: anime_idx array} and the
    parallel {user_idx: score array}, in file order. The scores feed the liked metrics."""
    t = pl.read_parquet(artifacts / f"eval_queries_{split}.parquet")
    qa: dict[int, list] = {}
    qs: dict[int, list] = {}
    for u, a, s in zip(t["user_idx"].to_numpy(), t["anime_idx"].to_numpy(), t["score"].to_numpy()):
        qa.setdefault(int(u), []).append(int(a))
        qs.setdefault(int(u), []).append(int(s))
    return ({u: np.asarray(v, dtype=np.int64) for u, v in qa.items()},
            {u: np.asarray(v, dtype=np.int64) for u, v in qs.items()})


def account_age_by_user(artifacts: Path = config.EXPORT) -> dict[int, float]:
    """user_idx -> account age (REF_YEAR - joined year; NaN if missing) via user_split + profiles.csv."""
    split = pl.read_parquet(artifacts / "user_split.parquet")
    prof = pd.read_csv(artifacts / "profiles.csv", usecols=["username", "joined"])
    yr = pd.to_datetime(prof["joined"], errors="coerce").dt.year
    age = dict(zip(prof["username"], (REF_YEAR - yr).astype(float)))
    return {int(u): age.get(name, float("nan"))
            for name, u in zip(split["username"], split["user_idx"])}
