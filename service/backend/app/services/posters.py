"""Posters plus fresh score/members from MAL v2, fetched in parallel with a file cache. Kept out of
the /recommend path so recommending stays fast; only needs the MAL client id.

Cards use this batch to show fresh numbers (matching MAL) instead of calling Jikan from the
browser (3 req/s, frequent 504s). No client id or a failed fetch -> poster None (the frontend
falls back to Jikan via <img onError>); score/members None -> the card keeps the backend snapshot.
Only successful entries are cached; a miss is retried next time.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "posters.json"
_LOCK = Lock()

# MAL v2 fields a card needs: picture, score, member count.
_FIELDS = "main_picture,mean,num_list_users"


def _load_cache() -> Dict[int, dict]:
    if CACHE_PATH.exists():
        try:
            raw = json.loads(CACHE_PATH.read_text())
        except (json.JSONDecodeError, ValueError, OSError):
            return {}
        # Only dict entries are valid; anything else (e.g. a bare URL string) is dropped and refetched.
        return {int(k): v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def _save_cache(cache: Dict[int, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


def _entry(data: dict) -> dict:
    pic = data.get("main_picture") or {}
    return {
        "poster": pic.get("large") or pic.get("medium"),
        "score": data.get("mean"),
        "members": data.get("num_list_users"),
    }


def fetch_posters(ids: List[int]) -> Dict[int, dict]:
    cache = _load_cache()
    missing = [i for i in dict.fromkeys(ids) if i not in cache]

    if missing:
        try:
            from app.clients.mal_api import get_anime_metadata  # lazy: needs MAL_CLIENT_ID
        except RuntimeError:
            get_anime_metadata = None  # no MAL_CLIENT_ID: the frontend falls back

        if get_anime_metadata is not None:
            def one(mal_id: int) -> Optional[dict]:
                data = get_anime_metadata(mal_id, fields=_FIELDS)
                return _entry(data) if data else None

            with ThreadPoolExecutor(max_workers=8) as ex:
                fetched = dict(zip(missing, ex.map(one, missing)))
            with _LOCK:
                cache = _load_cache()
                cache.update({k: v for k, v in fetched.items() if v})
                _save_cache(cache)

    empty = {"poster": None, "score": None, "members": None}
    return {i: cache.get(i, empty) for i in ids}
