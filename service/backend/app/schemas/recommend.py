"""Pydantic schemas: the I/O contract of POST /api/recommend.

AnimeItem matches Recommender._row (model/recommender.py) plus image_url, which the backend
never fills (cards load posters through /api/posters).
typing.Optional/List because the venv runs Python 3.9: pydantic cannot evaluate `X | None` there.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class AnimeItem(BaseModel):
    mal_id: int
    title: str
    type: str
    year: Optional[int] = None
    mal_score: Optional[float] = None
    popularity: Optional[int] = None   # MAL popularity rank (smaller = more popular), sorted client-side
    members: Optional[int] = None      # MAL users with the anime on their list, shown on the card
    start_date: Optional[str] = None   # "YYYY-MM-DD", for the client-side release-date sort
    pred: Optional[float] = None       # ranker score (main list only)
    cos: Optional[float] = None        # retriever cosine
    genres: List[str] = Field(default_factory=list)    # for client-side filters
    themes: List[str] = Field(default_factory=list)
    studios: List[str] = Field(default_factory=list)
    image_url: Optional[str] = None    # always null (see module docstring)


class RecommendRequest(BaseModel):
    username: Optional[str] = None
    mal_ids: Optional[List[int]] = None
    top_k: int = Field(default=20, ge=1, le=500)
    cold_k: int = Field(default=10, ge=0, le=500)
    live: bool = False                 # ignored: a username is always fetched live
    anchor_mal_id: Optional[int] = None  # "more like this mal_id", still personalized for the user
    sfw: bool = True                   # drop hentai from the results (on by default)
    exclude_ids: Optional[List[int]] = Field(default=None, max_length=1000)
    # exclude_ids: mal_ids marked "seen" in the UI, added to the seen mask (removed from the
    # results; not a like signal: likes go in mal_ids).
    # Neither username nor mal_ids -> guest: empty history (UserTower h_empty) -> generic,
    # popularity-leaning results, meta.source = "guest".


class RecommendMeta(BaseModel):
    source: str                        # live | mal_ids | guest | mock
    split: str = "-"                   # train/val/test for dataset users, else "-"
    history_count: int = 0             # positives fed to the model (those mapped to the corpus)
    total_entries: Optional[int] = None  # entries on the user's list (len of the animelist), for the UI
    alpha: Optional[float] = None      # ranker blend alpha
    k_retrieve: Optional[int] = None
    mode: str                          # mock | live


class RecommendResponse(BaseModel):
    main: List[AnimeItem]              # main list (LightGBM rerank)
    cold: List[AnimeItem]              # new anime (cold items, by cosine)
    meta: RecommendMeta
