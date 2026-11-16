"""Load the prepared train data, the example Dataset and the batch collate.

Histories are stored in full in users.parquet and kept ragged in RAM (values + offsets),
never padded to a full matrix. Training samples train_hist_len items per anchor each step
(augmentation); eval takes a prefix of eval_history_cap items (lists are sorted by score
desc, so the prefix is top-by-score). Also loads eval_seen (seen-mask) and the cold mask
(cold items are encoded with the OOV id at cold eval).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def load_feature_spec(train_data: Path) -> dict:
    with open(train_data / "feature_spec.json") as f:
        return json.load(f)


def load_logq(train_data: Path) -> torch.Tensor:
    """logq.npy, dense over anime_idx (PAD/OOV = -inf; every real item, cold ones included, is finite)."""
    arr = np.load(train_data / "logq.npy")
    return torch.from_numpy(arr.astype(np.float32))


def load_cold_mask(train_data: Path, num_items: int) -> torch.Tensor:
    """Bool [num_items], True = item in the cold set H (cold_items.parquet)."""
    idx = pq.read_table(train_data / "cold_items.parquet").column("anime_idx").to_numpy()
    mask = np.zeros(num_items, dtype=bool)
    mask[idx] = True
    return torch.from_numpy(mask)


def load_eval_seen(train_data: Path) -> Dict[int, np.ndarray]:
    """eval_seen.parquet -> {user_idx: sorted int64 array of seen_ids (every status)}."""
    t = pq.read_table(train_data / "eval_seen.parquet")
    uids = t.column("user_idx").to_numpy()
    col = t.column("seen_ids").combine_chunks()
    vals = col.values.to_numpy(zero_copy_only=False).astype(np.int64)
    offs = col.offsets.to_numpy().astype(np.int64)
    return {int(u): vals[offs[i]:offs[i + 1]] for i, u in enumerate(uids)}


def _pad_lists(col: List, width: int, dtype=np.int32) -> np.ndarray:
    """List[List[int]] -> [n, width] matrix padded with 0 (truncated beyond width)."""
    out = np.zeros((len(col), width), dtype=dtype)
    for i, lst in enumerate(col):
        if lst is None:
            continue
        v = np.asarray(lst, dtype=dtype)[:width]
        out[i, : len(v)] = v
    return out


class ItemTable:
    """Item features by anime_idx (row 0 = PAD, 1 = OOV, both neutral), as tensors the tower gathers."""

    CAT_COLS = [
        "type_id", "source_id", "rating_id", "demographics_id",
        "startyear_bucket", "episodes_bucket",
    ]

    def __init__(self, train_data: Path, synopsis_emb_file: str = "synopsis_emb.npy",
                 synopsis_low_info_file: str = "synopsis_low_info.npy",
                 use_synopsis: bool = False):
        t = pq.read_table(train_data / "item_features.parquet")
        d = t.to_pydict()
        self.num_items = len(d["anime_idx"])
        assert d["anime_idx"] == list(range(self.num_items)), "item_features must be dense over 0..N"

        self.cat = {c: torch.tensor(d[c], dtype=torch.long) for c in self.CAT_COLS}
        self.genres = torch.from_numpy(
            np.asarray([list(x) for x in d["genres_multihot"]], dtype=np.float32))
        self.themes = torch.from_numpy(
            np.asarray([list(x) for x in d["themes_multihot"]], dtype=np.float32))
        studio_lists = [list(x) if x is not None else [] for x in d["studio_ids"]]
        max_studios = max((len(x) for x in studio_lists), default=1) or 1
        self.studios = torch.from_numpy(_pad_lists(studio_lists, max_studios)).long()  # [N,S] pad 0

        # The frozen synopsis embedding loads only when use_synopsis is on and the artifact
        # exists; otherwise the attributes stay unset and nothing extra lands in RAM/GPU.
        emb_path = train_data / synopsis_emb_file
        if use_synopsis and emb_path.exists():
            self.synopsis_emb = torch.from_numpy(np.load(emb_path)).float()          # [N, raw_dim]
            self.synopsis_low_info = torch.from_numpy(
                np.load(train_data / synopsis_low_info_file).astype(bool))            # [N] bool
            assert len(self.synopsis_emb) == self.num_items, \
                f"synopsis_emb {len(self.synopsis_emb)} != num_items {self.num_items}"

    def to(self, device):
        self.cat = {k: v.to(device) for k, v in self.cat.items()}
        self.genres = self.genres.to(device)
        self.themes = self.themes.to(device)
        self.studios = self.studios.to(device)
        if hasattr(self, "synopsis_emb"):
            self.synopsis_emb = self.synopsis_emb.to(device)
            self.synopsis_low_info = self.synopsis_low_info.to(device)
        return self


class UserTable:
    """User features, full ragged history and hard negatives by user_idx.

    History is flat hist_vals/hist_scores plus hist_offs [U+1]: user u is
    vals[offs[u]:offs[u+1]], sorted by prep (score desc, ties asc), so any prefix is top-by-score.
    """

    def __init__(self, train_data: Path, hard_neg_cap: int):
        t = pq.read_table(train_data / "users.parquet")
        self.num_users = t.num_rows
        uidx = t.column("user_idx").to_numpy()
        assert (uidx == np.arange(self.num_users)).all(), "users must be dense over 0..U"

        self.split = np.asarray(t.column("split").to_pylist(), dtype=object)
        self.gender_id = t.column("gender_id").to_numpy().astype(np.int64)
        self.joined_bucket = t.column("joined_bucket").to_numpy().astype(np.int64)

        hcol = t.column("history_ids").combine_chunks()
        self.hist_vals = hcol.values.to_numpy(zero_copy_only=False).astype(np.int32)
        self.hist_offs = hcol.offsets.to_numpy().astype(np.int64)
        scol = t.column("history_scores").combine_chunks()
        self.hist_scores = scol.values.to_numpy(zero_copy_only=False).astype(np.int8)
        assert len(self.hist_vals) == len(self.hist_scores)
        assert len(self.hist_offs) == self.num_users + 1
        if len(self.hist_vals) == 0:                       # keep index 0 valid even with no history
            self.hist_vals = np.zeros(1, np.int32)
            self.hist_scores = np.zeros(1, np.int8)

        hn = t.column("hard_neg_ids").to_pylist()
        self.hardneg_pad = _pad_lists(hn, hard_neg_cap)        # [U,cap] pad 0
        self.hardneg_len = np.asarray(
            [min(len(x) if x is not None else 0, hard_neg_cap) for x in hn], dtype=np.int64,
        )

    def history_lens(self) -> np.ndarray:
        return self.hist_offs[1:] - self.hist_offs[:-1]

    def eval_history_batch(self, u: np.ndarray, cap: int):
        """Top-by-score history prefix for eval -> (ids [E,W] i64, mask [E,W] bool,
        scores [E,W] i64), W = min(longest in batch, cap) >= 1. Padding: id/score 0, mask False."""
        lens = np.minimum(self.hist_offs[u + 1] - self.hist_offs[u], cap)
        W = max(int(lens.max()) if len(lens) else 1, 1)
        ar = np.arange(W)[None, :]
        posn = np.minimum(ar, np.maximum(lens - 1, 0)[:, None])
        idx = self.hist_offs[u][:, None] + posn
        idx[lens == 0] = 0                                   # empty row: valid sentinel index
        ids = self.hist_vals[idx].astype(np.int64)
        scores = self.hist_scores[idx].astype(np.int64)
        mask = ar < lens[:, None]
        ids[~mask] = 0
        scores[~mask] = 0
        return ids, mask, scores


class ExamplesDataset(Dataset):
    """Positive (user_idx, anime_idx) pairs of one split (warm: train/val/test; cold: {val,test}_cold).

    subset keeps the first N (smoke). max_per_user caps examples per user per epoch: call
    resample(epoch) at the start of each epoch to redraw them (per user, without replacement).
    """

    def __init__(self, train_data: Path, split: str, subset: int = None,
                 max_per_user: int = None, seed: int = 42,
                 user_frac: float = None, user_frac_seed: int = 12345):
        t = pq.read_table(train_data / "examples" / f"split={split}" / "part-0.parquet")
        self.user_idx = t.column("user_idx").to_numpy()
        self.anime_idx = t.column("anime_idx").to_numpy()
        # score of the held-out query item: only for the liked metric at eval, never trained on
        self.score = t.column("score").to_numpy().astype(np.int64)
        if subset is not None:
            self.user_idx = self.user_idx[:subset]
            self.anime_idx = self.anime_idx[:subset]
            self.score = self.score[:subset]
        if user_frac is not None:
            self._keep_user_frac(user_frac, user_frac_seed)
        self.max_per_user = max_per_user
        self.seed = seed
        self.active = np.arange(len(self.user_idx))
        if max_per_user is not None:
            self.resample(0)

    def _keep_user_frac(self, frac: float, seed: int):
        """HP search: keep a random `frac` of distinct users (deterministic in seed). Random
        users rather than the first N keep the user/item distribution; only train examples are
        filtered, catalog/logQ/eval stay full. A lookup array keeps it O(n) on ~10M examples."""
        uids = np.unique(self.user_idx)
        rng = np.random.default_rng(seed)
        keep_u = uids[rng.random(len(uids)) < frac]
        keep = np.zeros(int(self.user_idx.max()) + 1, dtype=bool)
        keep[keep_u] = True
        m = keep[self.user_idx]
        self.user_idx = self.user_idx[m]
        self.anime_idx = self.anime_idx[m]
        self.score = self.score[m]

    def resample(self, epoch: int):
        """Redraw <= max_per_user examples per user (vectorized, deterministic in (seed, epoch))."""
        if self.max_per_user is None:
            return
        rng = np.random.default_rng(self.seed + 9973 * epoch)
        key = rng.random(len(self.user_idx)).astype(np.float32)
        order = np.lexsort((key, self.user_idx))             # by user, then random key
        uo = self.user_idx[order]
        idx = np.arange(len(uo))
        new_grp = np.r_[True, uo[1:] != uo[:-1]]
        start = np.maximum.accumulate(np.where(new_grp, idx, 0))
        rank = idx - start                                   # random rank within the user
        self.active = np.sort(order[rank < self.max_per_user])

    def __len__(self):
        return len(self.active)

    def __getitem__(self, i):
        j = self.active[i]
        return int(self.user_idx[j]), int(self.anime_idx[j])


class Collate:
    """Vectorized batch builder. Samples train_hist_len history items per anchor from the full
    list (with replacement when it is longer than L, accepting a few duplicates to stay
    vectorized; everything plus padding when shorter), removes the anchor, applies history
    dropout, samples m hard negatives per user (with a mask) and adds gender/joined.

    A class rather than a closure so it pickles and num_workers > 0 works under both macOS
    spawn and Linux fork. The RNG is seeded lazily from torch.initial_seed(), which differs
    per worker and per epoch.
    """

    def __init__(self, users: UserTable, hist_dropout: float, m_hardneg: int, train_hist_len: int):
        self.hist_vals = users.hist_vals
        self.hist_scores = users.hist_scores
        self.hist_offs = users.hist_offs
        self.hardneg_pad = users.hardneg_pad
        self.hardneg_len = users.hardneg_len
        self.gender_id = users.gender_id
        self.joined_bucket = users.joined_bucket
        self.col_idx = np.arange(users.hardneg_pad.shape[1])
        self.hist_dropout = hist_dropout
        self.m_hardneg = m_hardneg
        self.L = train_hist_len
        self._rng = None                                          # created in the worker, after pickling

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed())
        rng = self._rng
        u = np.fromiter((b[0] for b in batch), dtype=np.int64, count=len(batch))
        pos = np.fromiter((b[1] for b in batch), dtype=np.int64, count=len(batch))
        B = len(batch)

        # history: sample L positions from the user's full slice
        offs = self.hist_offs[u]                                   # [B]
        lens = self.hist_offs[u + 1] - offs                        # [B]
        L = self.L
        ar = np.arange(L)[None, :]
        long = lens > L
        rand_pos = (rng.random((B, L)) * np.maximum(lens, 1)[:, None]).astype(np.int64)
        take_pos = np.minimum(ar, np.maximum(lens - 1, 0)[:, None])
        posn = np.where(long[:, None], rand_pos, take_pos)         # [B,L]
        idx = offs[:, None] + posn
        idx[lens == 0] = 0                                         # empty row: valid sentinel index
        hist = self.hist_vals[idx].astype(np.int64)                # [B,L]
        hist_scores = self.hist_scores[idx].astype(np.int64)
        hist_mask = np.where(long[:, None], True, ar < lens[:, None])
        hist_mask &= lens[:, None] > 0
        # drop the anchor (sampling with replacement can hit it more than once)
        hist_mask &= hist != pos[:, None]
        # history dropout: some examples lose their whole history and fall back to h_empty
        if self.hist_dropout > 0:
            drop = rng.random(B) < self.hist_dropout
            hist_mask[drop] = False

        # hard negatives: m distinct items from the user's own pool (without replacement).
        # A pool smaller than m leaves PAD with mask False rather than inventing negatives.
        hn_lens = self.hardneg_len[u]                              # [B]
        hn_pool = self.hardneg_pad[u]                              # [B, cap]
        keys = rng.random((B, hn_pool.shape[1]))
        keys[self.col_idx[None, :] >= hn_lens[:, None]] = np.inf  # pad columns sort last
        chosen = np.argsort(keys, axis=1)[:, :self.m_hardneg]     # [B,m] random valid columns
        hn_ids = np.take_along_axis(hn_pool, chosen, axis=1)      # [B,m]
        hn_mask = np.arange(self.m_hardneg)[None, :] < hn_lens[:, None]  # [B,m]

        return {
            "user_idx": torch.from_numpy(u),
            "pos": torch.from_numpy(pos),
            "history_ids": torch.from_numpy(hist).long(),
            "history_mask": torch.from_numpy(hist_mask),
            "history_scores": torch.from_numpy(hist_scores).long(),
            "hardneg_ids": torch.from_numpy(hn_ids).long(),
            "hardneg_mask": torch.from_numpy(hn_mask),
            "gender_id": torch.from_numpy(self.gender_id[u]),
            "joined_bucket": torch.from_numpy(self.joined_bucket[u]),
        }
