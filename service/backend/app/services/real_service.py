"""RealService: loads the recommender once and serves recommendations for the web app.

It only talks to the model through model.base (UserHistory in, Recommendations out); which
architecture runs is the RECOMMENDER setting.

username -> always fetched live from MAL (animelist + profile).
mal_ids  -> used as they are (no MAL API needed).

Lazy imports, on purpose:
  - model.load() imports the architecture (torch/lightgbm) only when RealService is built, so
    mock mode never loads it.
  - mal_api is imported where it is used: it reads MAL_CLIENT_ID at import time and is only
    needed for live fetches, so mal_ids keeps working without a client id.
"""
from __future__ import annotations

from typing import List

from fastapi import HTTPException

from app.config import Settings
from app.schemas.anime import SearchResult
from app.schemas.recommend import (AnimeItem, RecommendMeta, RecommendRequest,
                                    RecommendResponse)
from app.services.base import RecommenderService
from app.services.details import media_label
import model
from model.base import UserHistory


class RealService(RecommenderService):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rec = model.load(settings.recommender)         # loads the bundle (a few seconds)
        self.model_loaded = True
        self._search_cache: dict = {}                       # normalized query -> results

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        if req.mal_ids or not req.username:                  # guest (neither given): empty history
            ids = req.mal_ids or []
            user = UserHistory(mal_ids=ids)
            total_entries = len(ids)
        else:                                                # username: always fetched live
            animelist, profile = self._fetch_live(req.username)
            if not animelist:                                # empty, unknown user, private list or network error
                raise HTTPException(
                    status_code=404,
                    detail=f"Can't get recommendations for '{req.username}' "
                           "(invalid username, private list, or connection error).",
                )
            user = UserHistory(username=req.username, animelist=animelist, profile=profile)
            total_entries = len(animelist)                   # every list entry (matches the profile)

        try:
            out = self.rec.recommend(user, top_k=req.top_k, cold_k=req.cold_k,
                                     exclude=req.exclude_ids or (),  # marked "seen" in the UI
                                     sfw=req.sfw, anchor=req.anchor_mal_id)
        except KeyError:
            raise HTTPException(
                status_code=422,
                detail=f"mal_id {req.anchor_mal_id} is not in the corpus.",
            )
        meta = {**out.meta, "total_entries": total_entries, "mode": "live"}
        if user.mal_ids == []:
            meta["source"] = "guest"
        return RecommendResponse(
            main=[AnimeItem(**r) for r in out.sections["main"]],
            cold=[AnimeItem(**r) for r in out.sections.get("cold", [])],
            meta=RecommendMeta(**meta),                      # keeps the fields the API defines
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
                in_corpus=self.rec.knows_anime(node["id"]),
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
