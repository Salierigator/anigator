"""Item feature table (export/twotower/details.csv) and assembly of the ranker feature matrix.

Shared by build_train, build_eval and serving, so train/eval/serve features match. Deterministic
given details.csv (same input -> same categorical codes); ranker_meta.json stores the feature
order and the categorical columns for the service.

Cold policy (no leak): rows flagged is_cold in item_index.parquet lose their maturity stats
(mal_score/scored_by/members/favorites/popularity/rank -> imputed as missing + flag), since a new
anime has none at serve time; content (genres/themes/type/...) plus episodes/recency stay
(published from day one). Applied inside ItemFeatures.load, so every path is consistent.

cross_features and user_stats_from_support compute the per-(user, candidate) columns from the
item vectors and the user's support; a single code path, so the candidate features at training
time match serving.

Reads only export/twotower/details.csv and export/twotower/item_index.parquet.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

from model.twotower.ranker import config
from model.twotower.retriever.inference import pad_lists

# Fixed column order: ranker.txt must match it exactly. Cross/user features are computed per
# (user, candidate) by cross_features; item features are gathered by anime_idx.
FEATURE_NAMES = [
    # cross (user x item)
    "cos_uv", "pool_rank", "hist_cos_max", "hist_cos_mean", "hist_cos_top5_mean",
    "genre_aff", "theme_aff", "genre_overlap", "score_gap",
    # user
    "u_n_rated", "u_mean_score", "u_std_score", "u_account_age", "support_len",
    # item numeric
    "mal_score", "mal_score_missing", "log_scored_by", "log_members", "log_favorites",
    "popularity", "rank", "rank_missing", "episodes", "recency_years",
    # item categorical (LightGBM native)
    "type_code", "source_code", "rating_code", "demo_code", "era_code",
]
CAT_COLS = ["type_code", "source_code", "rating_code", "demo_code", "era_code"]
N_CROSS = 14                                          # leading columns computed by the caller
ITEM_COLS = FEATURE_NAMES[N_CROSS:]                   # columns gathered by anime_idx
REF_YEAR = 2024                                       # reference year for recency and account age


def parse_list(s) -> list:
    if not isinstance(s, str) or not s.startswith("["):
        return []
    try:
        return list(ast.literal_eval(s))
    except (ValueError, SyntaxError):
        return []


def _codes(series: pd.Series) -> np.ndarray:
    """Categorical -> int code (sorted, NaN/missing -> 0). Deterministic given details.csv."""
    cat = series.astype("category")
    return (cat.cat.codes.to_numpy() + 1).astype(np.int32)   # +1: NaN (-1) -> 0, first category -> 1


def _multihot(lists: list[list], vocab: list[str]) -> np.ndarray:
    pos = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(lists), len(vocab)), dtype=np.float32)
    for r, tags in enumerate(lists):
        for t in tags:
            j = pos.get(t)
            if j is not None:
                out[r, j] = 1.0
    return out


class ItemFeatures:
    """Item feature arrays indexed by anime_idx (row == anime_idx, as in item_vectors)."""

    def __init__(self, item: dict, genres: np.ndarray, themes: np.ndarray, is_cold: np.ndarray):
        self.item = item                              # name -> [N] array
        self.genres = genres                          # [N, G] multihot
        self.themes = themes                          # [N, T] multihot
        self.is_cold = is_cold                        # [N] bool

    @classmethod
    def load(cls, artifacts: Path) -> "ItemFeatures":
        idx = pl.read_parquet(artifacts / "item_index.parquet").to_pandas()  # anime_idx, mal_id, is_cold
        det = pd.read_csv(artifacts / "details.csv", usecols=[
            "mal_id", "type", "source", "rating", "demographics", "score", "scored_by",
            "members", "favorites", "popularity", "rank", "episodes", "start_date",
            "genres", "themes"])
        df = idx.merge(det, on="mal_id", how="left").sort_values("anime_idx").reset_index(drop=True)
        cold = df["is_cold"].to_numpy(bool)

        # Cold policy: blank the maturity stats BEFORE imputing, so the medians come from warm items.
        df.loc[cold, ["score", "popularity", "rank"]] = np.nan
        df.loc[cold, ["scored_by", "members", "favorites"]] = 0

        year = pd.to_datetime(df["start_date"], errors="coerce", utc=True).dt.year
        era = pd.cut(year, bins=[-np.inf, 1989, 1999, 2009, 2017, np.inf], labels=False)

        item = {
            "mal_score": df["score"].fillna(df["score"].median()).to_numpy(np.float32),
            "mal_score_missing": df["score"].isna().to_numpy(np.float32),
            "log_scored_by": np.log1p(df["scored_by"].fillna(0).to_numpy(np.float32)),
            "log_members": np.log1p(df["members"].fillna(0).to_numpy(np.float32)),
            "log_favorites": np.log1p(df["favorites"].fillna(0).to_numpy(np.float32)),
            "popularity": df["popularity"].fillna(df["popularity"].median()).to_numpy(np.float32),
            "rank": df["rank"].fillna(df["rank"].median()).to_numpy(np.float32),
            "rank_missing": df["rank"].isna().to_numpy(np.float32),
            "episodes": df["episodes"].fillna(df["episodes"].median()).to_numpy(np.float32),
            "recency_years": (REF_YEAR - year.fillna(year.median())).to_numpy(np.float32),
            "type_code": _codes(df["type"]),
            "source_code": _codes(df["source"]),
            "rating_code": _codes(df["rating"]),
            "demo_code": _codes(df["demographics"].where(df["demographics"] != "[]")),
            "era_code": np.nan_to_num(era.to_numpy(np.float32), nan=-1).astype(np.int32) + 1,
        }
        gl = [parse_list(s) for s in df["genres"]]
        tl = [parse_list(s) for s in df["themes"]]
        gvocab = sorted({t for tags in gl for t in tags})
        tvocab = sorted({t for tags in tl for t in tags})
        return cls(item, _multihot(gl, gvocab), _multihot(tl, tvocab), cold)

    def affinity(self, cand_idx: np.ndarray, g_pref: np.ndarray, t_pref: np.ndarray):
        """Genre/theme affinity of the candidates with the user preference (mean multihot of the
        history). cand_idx [.., C], pref [.., G/T], broadcast over the batch."""
        gc, tc = self.genres[cand_idx], self.themes[cand_idx]              # [.., C, G/T]
        genre_aff = np.einsum("...cg,...g->...c", gc, g_pref)
        theme_aff = np.einsum("...ct,...t->...c", tc, t_pref)
        genre_overlap = np.einsum("...cg,...g->...c", gc, (g_pref > 0).astype(np.float32))
        return (genre_aff.astype(np.float32), theme_aff.astype(np.float32),
                genre_overlap.astype(np.float32))


def build_frame(itemfeat: ItemFeatures, cand_idx: np.ndarray, cross: dict) -> pd.DataFrame:
    """DataFrame[FEATURE_NAMES]: cross/user columns from `cross` (flat [n]) + item columns gathered
    by cand_idx."""
    data = {name: cross[name] for name in FEATURE_NAMES[:N_CROSS]}
    for name in ITEM_COLS:
        data[name] = itemfeat.item[name][cand_idx]
    return pd.DataFrame(data)[FEATURE_NAMES]


def cross_features(V: np.ndarray, itemfeat: ItemFeatures, cand: np.ndarray, cos: np.ndarray,
                   hist_feat: list[np.ndarray], user_stats: dict[str, np.ndarray]) -> dict:
    """Cross features [n, D] -> {name: flat [n*D]} in FEATURE_NAMES[:N_CROSS] order. hist_feat is
    the support (top-by-score first); only its first HIST_FEAT_CAP items are used, for features only."""
    n, D = cand.shape
    hp, hm = pad_lists(hist_feat, config.HIST_FEAT_CAP)          # [n, H]
    Vc = torch.from_numpy(V[cand])                                # [n, D, d]
    Vh = torch.from_numpy(V[hp]) * torch.from_numpy(hm)[..., None]  # pad -> 0 vec
    sims = torch.bmm(Vc, Vh.transpose(1, 2))                      # [n, D, H]
    sims.masked_fill_(~torch.from_numpy(hm)[:, None, :], float("-inf"))
    hist_max = sims.max(dim=2).values
    hist_max = torch.where(torch.isfinite(hist_max), hist_max, torch.zeros(()))
    top5 = sims.topk(min(5, sims.shape[2]), dim=2).values
    lens = torch.from_numpy(hm.sum(1, dtype=np.float32))          # [n]
    top5_mean = torch.where(torch.isfinite(top5), top5, torch.zeros(())).sum(2) \
        / torch.minimum(lens, torch.tensor(5.0)).clamp(min=1)[:, None]
    hist_mean = torch.where(torch.isfinite(sims), sims, torch.zeros(())).sum(2) \
        / lens.clamp(min=1)[:, None]

    counts = hm.sum(1, keepdims=True).astype(np.float32).clip(min=1)
    g_pref = (itemfeat.genres[hp] * hm[..., None]).sum(1) / counts   # [n, G]
    t_pref = (itemfeat.themes[hp] * hm[..., None]).sum(1) / counts
    ga, ta, go = itemfeat.affinity(cand, g_pref, t_pref)             # [n, D]

    mal = itemfeat.item["mal_score"][cand]                           # [n, D]
    rep = lambda a: np.repeat(a.astype(np.float32), D)
    return {
        "cos_uv": cos.ravel(),
        "pool_rank": np.tile(np.arange(D, dtype=np.float32), n),
        "hist_cos_max": hist_max.numpy().ravel(),
        "hist_cos_mean": hist_mean.numpy().ravel(),
        "hist_cos_top5_mean": top5_mean.numpy().ravel(),
        "genre_aff": ga.ravel(), "theme_aff": ta.ravel(), "genre_overlap": go.ravel(),
        "score_gap": (mal - user_stats["u_mean_score"][:, None]).astype(np.float32).ravel(),
        "u_n_rated": rep(user_stats["u_n_rated"]),
        "u_mean_score": rep(user_stats["u_mean_score"]),
        "u_std_score": rep(user_stats["u_std_score"]),
        "u_account_age": rep(user_stats["u_account_age"]),
        "support_len": rep(user_stats["support_len"]),
    }


def user_stats_from_support(supp_scores: list[np.ndarray], ages: np.ndarray) -> dict:
    """User stats from the support scores (rated = score >= 1; score 0 = completed without a score)."""
    n_rated, mean_s, std_s, slen = [], [], [], []
    for sc in supp_scores:
        rated = sc[sc >= 1]
        n_rated.append(float(len(rated)))
        mean_s.append(float(rated.mean()) if len(rated) else 0.0)
        std_s.append(float(rated.std()) if len(rated) else 0.0)
        slen.append(float(len(sc)))
    return {
        "u_n_rated": np.asarray(n_rated, np.float32),
        "u_mean_score": np.asarray(mean_s, np.float32),
        "u_std_score": np.asarray(std_s, np.float32),
        "u_account_age": ages.astype(np.float32),
        "support_len": np.asarray(slen, np.float32),
    }
