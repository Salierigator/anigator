"""Common interface of the recommendation services: MockService and RealService implement it.

New backend logic (e.g. recommending for several users, by genre...) = a new method here,
implemented in both classes; routes only call through the interface, so they never depend on
the model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.schemas.anime import SearchResult
from app.schemas.recommend import RecommendRequest, RecommendResponse


class RecommenderService(ABC):
    model_loaded: bool = False         # read by the health endpoint

    @abstractmethod
    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        ...

    @abstractmethod
    def search(self, q: str, limit: int) -> List[SearchResult]:
        """Title autocomplete -> candidates for the guest picker (GET /api/search)."""
        ...
