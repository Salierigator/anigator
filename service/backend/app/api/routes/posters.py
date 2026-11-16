"""POST /api/posters: {ids: [...]} -> {posters: {mal_id: {poster, score, members}}}.

Kept apart from /recommend so recommending stays fast. The backend fetches MAL v2 (in parallel,
cached); the frontend falls back to Jikan on a miss.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.posters import PostersRequest, PostersResponse
from app.services.posters import fetch_posters

router = APIRouter()


@router.post("/posters", response_model=PostersResponse)
def posters(req: PostersRequest) -> PostersResponse:
    return PostersResponse(posters=fetch_posters(req.ids))
