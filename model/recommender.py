"""Recommender: the serving core (retrieve, rerank, cold section).

Used by the CLI (service/backend/recommend.py) and the API (service/backend/app/services/
real_service.py). Reads only export/.

Import order matters: torch before lightgbm (two OpenMP runtimes segfault on macOS).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import torch                                               # before lightgbm

from model.paths import EXPORT
from model.ranker.features import REF_YEAR, FEATURE_NAMES, ItemFeatures, build_frame, _parse_list
from model.ranker.pool import cross_features, encode_users, topk_pool, user_stats_from_support
from model.ranker.user_encode import encode_gender_joined, load_user_encoder
import lightgbm as lgb                                     # after torch


class Recommender:
    """Loads the bundle once (a few seconds); recommend() is then cheap and reusable."""

    def __init__(self, device: str = "cpu"):
        self.enc, self.meta = load_user_encoder(device)
        self.cap = self.meta.get("eval_history_cap", 1024)
        self.V = self.enc.item_cache.numpy()
        self.itemfeat = ItemFeatures.load(EXPORT)
        self.cold_idx = np.flatnonzero(self.itemfeat.is_cold)
        self.warm_idx = np.flatnonzero(~self.itemfeat.is_cold)
        self.booster = lgb.Booster(model_file=str(EXPORT / "ranker.txt"))
        assert self.booster.feature_name() == FEATURE_NAMES, "ranker.txt feature order differs from FEATURE_NAMES"
        rmeta = json.loads((EXPORT / "ranker_meta.json").read_text())
        self.alpha = float(rmeta["blend_alpha"])
        self.k_retrieve = int(rmeta["k_retrieve"])
        self.newest = len(self.meta["user_features"]["joined"]["bins"]) - 2

        idx = pl.read_parquet(EXPORT / "item_index.parquet")
        self.idx2mal = idx["mal_id"].to_numpy()
        real = idx.filter(pl.col("anime_idx") >= 2)
        self.mal2idx = dict(zip(real["mal_id"].to_list(), real["anime_idx"].to_list()))
        det = pd.read_csv(EXPORT / "details.csv",
                          usecols=["mal_id", "title", "type", "score", "start_date",
                                   "popularity", "members", "genres", "themes",
                                   "studios", "rating"])
        det["year"] = pd.to_datetime(det["start_date"], errors="coerce", utc=True).dt.year
        det["start_date"] = det["start_date"].str.slice(0, 10)  # "YYYY-MM-DD", sorted client-side
        for col in ("genres", "themes", "studios"):        # str -> list[str], for client-side filters
            det[col] = det[col].map(_parse_list)
        self.detail = det.set_index("mal_id")[["title", "type", "score", "year",
                                               "start_date", "popularity", "members",
                                               "genres", "themes", "studios"]]
        # SFW: anime_idx of hentai items (genre 'Hentai' or rating 'Rx - Hentai')
        is_hentai = (det["rating"] == "Rx - Hentai") | \
            det["genres"].map(lambda g: "Hentai" in g)
        self.nsfw_idx = np.asarray(
            [self.mal2idx[m] for m in det.loc[is_hentai, "mal_id"] if m in self.mal2idx],
            dtype=np.int64)

    def user_from_dataset(self, username: str) -> dict | None:
        """A username from user_split -> history/gender/joined from users_history (offline)."""
        row = pl.scan_parquet(EXPORT / "user_split.parquet") \
            .filter(pl.col("username") == username).collect()
        if row.height == 0:
            return None
        uidx = int(row["user_idx"][0])
        uh = pl.scan_parquet(EXPORT / "users_history.parquet") \
            .filter(pl.col("user_idx") == uidx).collect()
        prof = pd.read_csv(EXPORT / "profiles.csv", usecols=["username", "joined"])
        yr = pd.to_datetime(
            prof.loc[prof["username"] == username, "joined"], errors="coerce").dt.year
        hist = np.asarray(uh["history_ids"][0].to_list(), dtype=np.int64)
        scores = np.asarray(uh["history_scores"][0].to_list(), dtype=np.int64)
        hard_neg = np.asarray(uh["hard_neg_ids"][0].to_list(), dtype=np.int64)
        return {
            "hist_idx": hist, "hist_score": scores,
            "seen": np.union1d(hist, hard_neg),
            "gender_id": int(uh["gender_id"][0]), "joined_bucket": int(uh["joined_bucket"][0]),
            "age": float(REF_YEAR - yr.iloc[0]) if len(yr) and pd.notna(yr.iloc[0]) else np.nan,
            "split": str(row["split"][0]), "source": "dataset",
        }

    def user_from_animelist(self, animelist: list, profile: dict | None,
                            username: str) -> dict:
        """Live MAL v2 animelist JSON -> positives (status in {completed, watching} and score not in
        [1, 4], as in the retriever's prep), sorted by score desc; seen = every mapped anime, any status."""
        seen, pos = set(), {}
        for e in animelist:
            ai = self.mal2idx.get((e.get("node") or {}).get("id"))
            if ai is None:                                 # not in the catalog
                continue
            seen.add(ai)
            ls = e.get("list_status") or {}
            score = ls.get("score") or 0
            if ls.get("status") in ("completed", "watching") and score not in (1, 2, 3, 4):
                pos[ai] = max(score, pos.get(ai, 0))
        items = sorted(pos.items(), key=lambda kv: (-kv[1], kv[0]))
        gender_id, joined_bucket, age = 0, self.newest, np.nan
        if profile:
            df = pd.DataFrame([{"username": username, "gender": profile.get("gender"),
                                "joined": profile.get("joined")}])
            enc = encode_gender_joined(df, self.meta).iloc[0]
            gender_id, joined_bucket = int(enc["gender_id"]), int(enc["joined_bucket"])
            yr = pd.to_datetime(profile.get("joined"), errors="coerce", utc=True)
            age = float(REF_YEAR - yr.year) if pd.notna(yr) else np.nan
        return {
            "hist_idx": np.asarray([a for a, _ in items], dtype=np.int64),
            "hist_score": np.asarray([s for _, s in items], dtype=np.int64),
            "seen": np.asarray(sorted(seen), dtype=np.int64),
            "gender_id": gender_id, "joined_bucket": joined_bucket, "age": age,
            "split": "-", "source": "live",
        }

    def user_from_mal_ids(self, ids: list[int]) -> dict:
        """A list of mal_ids -> treated as completed without a score (score 0 is still a valid positive)."""
        hist = np.asarray(sorted({self.mal2idx[i] for i in ids if i in self.mal2idx}),
                          dtype=np.int64)
        return {"hist_idx": hist, "hist_score": np.zeros(len(hist), dtype=np.int64),
                "seen": hist, "gender_id": 0, "joined_bucket": self.newest,
                "age": np.nan, "split": "-", "source": "mal_ids"}

    def encode_U(self, user: dict) -> torch.Tensor:
        """U [1, d] for a user dict."""
        return encode_users(self.enc, [user["hist_idx"]], [user["hist_score"]],
                            np.asarray([user["gender_id"]]), np.asarray([user["joined_bucket"]]),
                            self.cap)

    def recommend(self, user: dict, top_k: int = 20, cold_k: int = 10,
                  anchor_mal_id: int | None = None, sfw: bool = True) -> dict:
        U = self.encode_U(user)
        # SFW: hentai items join the mask, so they leave both the warm and the cold pool before ranking.
        block = np.union1d(user["seen"], self.nsfw_idx) if sfw else user["seen"]
        # The pool may be deeper than k_retrieve: item features do not depend on depth (pool_rank is
        # the absolute cosine rank), so the client can filter and slice.
        depth = max(self.k_retrieve, top_k)
        if anchor_mal_id is None:
            mask = [block]
            cold_query = U
            # Main list: warm candidates only, reranked by LightGBM (cold items never go through it).
            cand, cos = topk_pool(U, self.enc.item_cache, mask, depth,
                                  cold_idx=self.cold_idx)
        else:
            # "More like X": the pool comes from the anchor, but cos_uv is recomputed user-item so
            # the ranker sees the distribution it was trained on.
            aidx = self.mal2idx.get(int(anchor_mal_id))
            if aidx is None:
                raise KeyError(anchor_mal_id)
            cold_query = self.enc.item_cache[aidx:aidx + 1]    # [1, d]: the vector of X
            mask = [np.union1d(block, [aidx])]                 # exclude X itself
            cand, _ = topk_pool(cold_query, self.enc.item_cache, mask, depth,
                                cold_idx=self.cold_idx)
            cos = (U @ self.enc.item_cache[torch.from_numpy(cand[0])].t()).numpy()  # [1, k]
        stats = user_stats_from_support([user["hist_score"]],
                                        np.asarray([user["age"]], dtype=np.float64))
        cross = cross_features(self.V, self.itemfeat, cand, cos, [user["hist_idx"]], stats)
        X = build_frame(self.itemfeat, cand.ravel(), cross)
        pred = self.booster.predict(X)
        if self.alpha < 1.0:                               # alpha = 1: sort by pred directly
            from model.ranker.metrics import blend
            score = blend(cos.ravel().astype(np.float64), pred,
                          np.array([0, cand.shape[1]]), self.alpha)
        else:
            score = pred
        order = np.argsort(-score)[:top_k]
        main = [self._row(int(cand[0, j]), pred=float(pred[j]), cos=float(cos[0, j]))
                for j in order]
        # Cold section: cold items by cosine (warm and seen masked); in anchor mode, cold items like X.
        ccand, ccos = topk_pool(cold_query, self.enc.item_cache, mask, cold_k,
                                cold_idx=self.warm_idx)
        cold = [self._row(int(a), cos=float(c)) for a, c in zip(ccand[0], ccos[0])]
        return {"main": main, "cold": cold}

    def _row(self, anime_idx: int, **scores) -> dict:
        mal_id = int(self.idx2mal[anime_idx])
        d = self.detail.loc[mal_id] if mal_id in self.detail.index else None
        return {
            "mal_id": mal_id,
            "title": str(d["title"]) if d is not None else "?",
            "type": str(d["type"]) if d is not None and pd.notna(d["type"]) else "?",
            "year": int(d["year"]) if d is not None and pd.notna(d["year"]) else None,
            "mal_score": round(float(d["score"]), 2)
                         if d is not None and pd.notna(d["score"]) else None,
            "popularity": int(d["popularity"])
                          if d is not None and pd.notna(d["popularity"]) else None,
            "members": int(d["members"])
                       if d is not None and pd.notna(d["members"]) else None,
            "start_date": str(d["start_date"])
                          if d is not None and pd.notna(d["start_date"]) else None,
            "genres": list(d["genres"]) if d is not None else [],
            "themes": list(d["themes"]) if d is not None else [],
            "studios": list(d["studios"]) if d is not None else [],
            **{k: round(v, 4) for k, v in scores.items()},
        }
