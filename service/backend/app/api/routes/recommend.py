"""POST /api/recommend: username or mal_ids -> {main, cold, meta}."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_service
from app.schemas.recommend import RecommendRequest, RecommendResponse
from app.services.base import RecommenderService

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
def recommend(
    req: RecommendRequest, service: RecommenderService = Depends(get_service)
) -> RecommendResponse:
    return service.recommend(req)
