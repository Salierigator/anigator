"""Schemas for POST /api/posters: batched poster + fresh score/members by mal_id (MAL v2)."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PostersRequest(BaseModel):
    ids: List[int] = Field(..., min_length=1, max_length=300)


class PosterEntry(BaseModel):
    poster: Optional[str] = None    # image URL (null if unavailable)
    score: Optional[float] = None   # latest MAL mean score (null if not scored yet)
    members: Optional[int] = None   # num_list_users: users with the anime on their list


class PostersResponse(BaseModel):
    posters: Dict[int, PosterEntry]  # mal_id -> {poster, score, members}
