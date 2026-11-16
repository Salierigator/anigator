"""MockService: serves the JSON fixture without loading the model.

Lets the frontend run without torch/lightgbm/export/. Reads fixtures/recommend_sample.json once at
startup and cuts it to the request's top_k/cold_k.
"""
from __future__ import annotations

import json
from typing import List

from app.config import Settings
from app.schemas.anime import SearchResult
from app.schemas.recommend import (AnimeItem, RecommendMeta, RecommendRequest,
                                    RecommendResponse)
from app.services.base import RecommenderService


class MockService(RecommenderService):
    model_loaded = False

    def __init__(self, settings: Settings):
        path = settings.fixtures_dir / "recommend_sample.json"
        self._data = json.loads(path.read_text(encoding="utf-8"))

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        drop = set(req.exclude_ids or [])                    # mimics the seen mask
        main = [AnimeItem(**r) for r in self._data["main"]
                if r["mal_id"] not in drop][: req.top_k]
        cold = [AnimeItem(**r) for r in self._data["cold"]
                if r["mal_id"] not in drop][: req.cold_k]
        fmeta = self._data.get("meta", {})
        meta = RecommendMeta(
            source="mock",
            split=fmeta.get("split", "-"),
            history_count=fmeta.get("history_count", 0),
            total_entries=fmeta.get("total_entries"),
            alpha=fmeta.get("alpha"),
            k_retrieve=fmeta.get("k_retrieve"),
            mode="mock",
        )
        return RecommendResponse(main=main, cold=cold, meta=meta)

    def search(self, q: str, limit: int) -> List[SearchResult]:
        # substring match over the fixture items (main + cold), so the picker works offline
        ql = q.lower()
        hits = [r for r in self._data["main"] + self._data["cold"]
                if ql in r["title"].lower()]
        return [SearchResult(mal_id=r["mal_id"], title=r["title"], type=r.get("type"),
                             year=r.get("year"), mal_score=r.get("mal_score"))
                for r in hits[:limit]]
