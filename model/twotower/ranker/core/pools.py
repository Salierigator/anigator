"""The pool parquet format: PoolWriter writes it, load_pool_arrays reads it back.

Rows are sorted by contiguous qid, one group per user, each group sorted by cosine desc (pool_rank
ascending). A side table holds one row per group (qid, user_idx, r_total, r_liked, U, hist_top64).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from model.twotower.ranker import config
from model.twotower.ranker.core.blend import group_offsets


class PoolWriter:
    """Writes the pool parquet chunk by chunk (never holds every row in RAM) plus a per-user side table."""

    def __init__(self, pool_path: Path, users_path: Path):
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        self.pool_path, self.users_path = pool_path, users_path
        self.writer = None
        self.users_rows = {"qid": [], "user_idx": [], "r_total": [], "r_liked": [],
                           "U": [], "hist_top64": []}
        self.next_qid = 0

    def add_chunk(self, uids: np.ndarray, cand: np.ndarray, labels: np.ndarray,
                  frame: pd.DataFrame, U: np.ndarray, hist_top64: list[np.ndarray],
                  r_total: np.ndarray, label_liked: np.ndarray | None = None,
                  r_liked: np.ndarray | None = None, keep: np.ndarray | None = None,
                  target_score: np.ndarray | None = None):
        """cand/labels/label_liked [n, D]; frame = build_frame output, flat [n*D]; keep = bool [n]
        (groups to write). label_liked = candidate in the liked queries (binary); r_liked [n] =
        liked queries per user, including those outside the pool. target_score [n, D] = raw score
        of a target candidate (else 0), so train_lgbm --relabel needs no rebuild. None -> a zeros
        column, keeping a single schema."""
        n, D = cand.shape
        if label_liked is None:
            label_liked = np.zeros_like(labels)
        if r_liked is None:
            r_liked = np.zeros(n, dtype=np.int64)
        if target_score is None:
            target_score = np.zeros_like(labels)
        if keep is None:
            keep = np.ones(n, dtype=bool)
        rows = np.repeat(keep, D)
        qid_local = np.cumsum(keep) - 1                            # dense within the chunk
        qid = (self.next_qid + qid_local).astype(np.int64)

        out = frame[rows].reset_index(drop=True)
        out.insert(0, "qid", np.repeat(qid, D)[rows])
        out.insert(1, "user_idx", np.repeat(uids.astype(np.int64), D)[rows])
        out.insert(2, "anime_idx", cand.ravel()[rows])
        out["label"] = labels.ravel()[rows]
        out["label_liked"] = label_liked.ravel()[rows]
        out["target_score"] = target_score.ravel()[rows]
        table = pa.Table.from_pandas(out, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.pool_path, table.schema, compression="zstd")
        self.writer.write_table(table)

        for i in np.flatnonzero(keep):
            self.users_rows["qid"].append(self.next_qid + int(qid_local[i]))
            self.users_rows["user_idx"].append(int(uids[i]))
            self.users_rows["r_total"].append(int(r_total[i]))
            self.users_rows["r_liked"].append(int(r_liked[i]))
            self.users_rows["U"].append(U[i].astype(np.float32))
            self.users_rows["hist_top64"].append(hist_top64[i][:config.HIST_TOP64].astype(np.int32))
        self.next_qid += int(keep.sum())

    def close(self):
        if self.writer is not None:
            self.writer.close()
        pl.DataFrame({
            "qid": np.asarray(self.users_rows["qid"], np.int64),
            "user_idx": np.asarray(self.users_rows["user_idx"], np.int64),
            "r_total": np.asarray(self.users_rows["r_total"], np.int32),
            "r_liked": np.asarray(self.users_rows["r_liked"], np.int32),
            "U": [u.tolist() for u in self.users_rows["U"]],
            "hist_top64": [h.tolist() for h in self.users_rows["hist_top64"]],
        }).write_parquet(self.users_path)
        return self.next_qid


def load_pool_arrays(pool_path, users_path, k: int, max_groups: int | None = None):
    """Read a pool parquet, keeping the top k of each group by pool_rank. Returns
    (df, cos f64, label i8, offsets, r_total, users). max_groups keeps only qid < max_groups (smoke)."""
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
