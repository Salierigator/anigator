"""The contract between a recommender architecture and whatever serves it.

Everything here is in MAL terms (mal_id, a MAL animelist, a MAL profile). How an architecture
indexes, encodes or ranks is its own business and stays inside model/<name>/.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable


@dataclass
class UserHistory:
    """One user as MAL describes them. The history comes from `mal_ids` when given, else from
    `animelist`; with neither, the recommender looks `username` up in its own data."""
    username: str | None = None
    animelist: list[dict] | None = None    # MAL v2 animelist entries: {"node": ..., "list_status": ...}
    profile: dict | None = None            # MAL profile (gender, joined, ...)
    mal_ids: list[int] | None = None       # bare picks, without scores


@dataclass
class Recommendations:
    sections: dict[str, list[dict]]        # "main" always; other sections depend on the architecture
    meta: dict                             # facts about this call (source, history_count, ...)


class Recommender(ABC):
    """A loaded model. Rows in a section are dicts with `mal_id`, the display fields of the anime
    and the architecture's own scores."""

    @abstractmethod
    def recommend(self, user: UserHistory, *, top_k: int = 20, cold_k: int = 10,
                  exclude: Iterable[int] = (), sfw: bool = True,
                  anchor: int | None = None) -> Recommendations:
        """Top `top_k` rows in "main"; `cold_k` sizes the "cold" section of architectures that
        have one. `exclude` are mal_ids to leave out on top of the user's own list. `anchor` asks
        for anime like that mal_id, still personalized.

        Raises KeyError when `anchor` is not in the corpus and LookupError when the user has no
        history source and the username is unknown."""

    @abstractmethod
    def knows_anime(self, mal_id: int) -> bool:
        """Whether the model can use and recommend this anime."""

    def knows_user(self, username: str) -> bool:
        """Whether a bare username (no animelist, no mal_ids) is enough for recommend()."""
        return False
