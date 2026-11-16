"""RealService: loads the Recommender once and serves recommendations for the web app.

username -> always fetched live from MAL (animelist + profile) -> user_from_animelist.
mal_ids  -> user_from_mal_ids (no MAL API needed).

Lazy imports, on purpose:
  - Recommender is imported inside __init__, so mock mode never loads torch/lightgbm.
  - mal_api is imported where it is used: it reads MAL_CLIENT_ID at import time and is only
    needed for live fetches, so mal_ids keeps working without a client id.
"""
from __future__ import annotations

from typing import List

import numpy as np
from fastapi import HTTPException

from app.config import Settings
from app.schemas.anime import SearchResult
from app.schemas.recommend import (AnimeItem, RecommendMeta, RecommendRequest,
                                    RecommendResponse)
from app.services.base import RecommenderService
from app.services.details import media_label


class RealService(RecommenderService):
    def __init__(self, settings: Settings):
        self.settings = settings
        from model.recommender import Recommender           # lazy: torch/lightgbm
        self.rec = Recommender()                            # loads the bundle (a few seconds)
        self.model_loaded = True
        self._search_cache: dict = {}                       # normalized query -> results

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        if req.mal_ids or not req.username:                  # guest (neither given): empty history
            ids = req.mal_ids or []
            user = self.rec.user_from_mal_ids(ids)
            if not ids:
                user["source"] = "guest"
            total_entries = len(ids)
        else:                                                # username: always fetched live
            animelist, profile = self._fetch_live(req.username)
            if not animelist:                                # empty, unknown user, private list or network error
                raise HTTPException(
                    status_code=404,
                    detail=f"Can't get recommendations for '{req.username}' "
                           "(invalid username, private list, or connection error).",
                )
            user = self.rec.user_from_animelist(animelist, profile, req.username)
            total_entries = len(animelist)                   # every list entry (matches the profile)

        if req.exclude_ids:                                  # marked "seen" in the UI: masked too
            extra = [self.rec.mal2idx[i] for i in req.exclude_ids
                     if i in self.rec.mal2idx]
            user["seen"] = np.union1d(user["seen"], np.asarray(extra, dtype=np.int64))

        try:
            out = self.rec.recommend(user, top_k=req.top_k, cold_k=req.cold_k,
                                     anchor_mal_id=req.anchor_mal_id, sfw=req.sfw)
        except KeyError:
            raise HTTPException(
                status_code=422,
                detail=f"mal_id {req.anchor_mal_id} is not in the corpus.",
            )
        return RecommendResponse(
            main=[AnimeItem(**r) for r in out["main"]],
            cold=[AnimeItem(**r) for r in out["cold"]],
            meta=RecommendMeta(
                source=user["source"], split=user["split"],
                history_count=len(user["hist_idx"]), total_entries=total_entries,
                alpha=self.rec.alpha, k_retrieve=self.rec.k_retrieve, mode="live",
            ),
        )

    def search(self, q: str, limit: int) -> List[SearchResult]:
        """Proxies MAL v2 search (it matches English titles and synonyms; details.csv only has
        romaji) and flags in_corpus, so the frontend can tell which picks the model does not know."""
        key = (q.lower().strip(), limit)
        if key in self._search_cache:
            return self._search_cache[key]
        try:
            from app.clients import mal_api                  # lazy: reads MAL_CLIENT_ID on import
        except RuntimeError:                                 # MAL_CLIENT_ID missing
            raise HTTPException(status_code=503, detail="search is unavailable")
        out = []
        for node in mal_api.search_anime(q, limit=limit):
            pic = node.get("main_picture") or {}
            start = node.get("start_date") or ""
            out.append(SearchResult(
                mal_id=node["id"], title=node.get("title") or "?",
                title_english=(node.get("alternative_titles") or {}).get("en") or None,
                type=media_label(node.get("media_type")),
                year=int(start[:4]) if start[:4].isdigit() else None,
                mal_score=node.get("mean"),
                image_url=pic.get("medium") or pic.get("large"),
                in_corpus=node["id"] in self.rec.mal2idx,
            ))
        if out:                                              # errors/empty results stay uncached (retryable)
            if len(self._search_cache) > 256:
                self._search_cache.clear()
            self._search_cache[key] = out
        return out

    def _fetch_live(self, username: str):
        try:
            from app.clients import mal_api                  # lazy: reads MAL_CLIENT_ID on import
        except RuntimeError:                                 # MAL_CLIENT_ID missing
            raise HTTPException(status_code=503, detail="username lookup is unavailable")
        animelist = mal_api.get_user_anime_list(username)
        profile = mal_api.get_user_profile(username)
        return animelist, profile
