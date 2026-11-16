"""Schemas for GET /api/anime/{mal_id} (details modal) and GET /api/search (guest picker).
typing.Optional/List because the venv runs Python 3.9."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AnimeDetails(BaseModel):
    mal_id: int
    title: str
    title_english: Optional[str] = None
    image_url: Optional[str] = None
    score: Optional[float] = None      # MAL mean score
    rank: Optional[int] = None
    popularity: Optional[int] = None
    type: Optional[str] = None         # TV | Movie | OVA | ONA | Special | Music | ...
    year: Optional[int] = None
    episodes: Optional[int] = None
    status: Optional[str] = None       # "Finished Airing" | "Currently Airing" | "Not Yet Aired"
    genres: List[str] = Field(default_factory=list)   # MAL merges genres, themes and demographics
    studios: List[str] = Field(default_factory=list)
    synopsis: Optional[str] = None


class SearchResult(BaseModel):
    mal_id: int
    title: str
    title_english: Optional[str] = None
    type: Optional[str] = None
    year: Optional[int] = None
    mal_score: Optional[float] = None
    image_url: Optional[str] = None
    in_corpus: bool = True             # False = unknown to the model (the pick is ignored)


class SearchResponse(BaseModel):
    results: List[SearchResult]
