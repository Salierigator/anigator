"""Details of one anime for the frontend modal: MAL v2 + a file cache (same pattern as posters.py).

Proxied by the backend rather than fetched from Jikan by the browser: Jikan allows 3 req/s per IP
and the modal would queue behind the poster fallbacks of hundreds of cards. MAL v2 with a client
id has no such tight limit, and details barely change, so the cache answers instantly from the
second call. Only successful fetches are cached, so a failure is retried next time.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "details.json"
_LOCK = Lock()

DETAIL_FIELDS = (
    "id,title,alternative_titles,main_picture,mean,rank,popularity,media_type,"
    "status,num_episodes,start_date,start_season,genres,studios,synopsis"
)

_MEDIA_TYPE = {"tv": "TV", "movie": "Movie", "ova": "OVA", "ona": "ONA",
               "special": "Special", "tv_special": "TV Special", "music": "Music",
               "cm": "CM", "pv": "PV"}


def media_label(media: Optional[str]) -> Optional[str]:
    """MAL v2 media_type -> display label (shared by details and search)."""
    return _MEDIA_TYPE.get(media, media.upper() if media else None)


def _load_cache() -> Dict[int, dict]:
    if CACHE_PATH.exists():
        try:
            return {int(k): v for k, v in json.loads(CACHE_PATH.read_text()).items()}
        except (json.JSONDecodeError, ValueError, OSError):
            return {}
    return {}


def _map_node(data: dict) -> dict:
    """MAL v2 JSON -> the AnimeDetails shape (schemas/anime.py)."""
    pic = data.get("main_picture") or {}
    season = data.get("start_season") or {}
    start_date = data.get("start_date") or ""
    year = season.get("year") or (int(start_date[:4]) if start_date[:4].isdigit() else None)
    media = data.get("media_type")
    status = data.get("status")
    return {
        "mal_id": data["id"],
        "title": data.get("title") or "?",
        "title_english": (data.get("alternative_titles") or {}).get("en") or None,
        "image_url": pic.get("large") or pic.get("medium"),
        "score": data.get("mean"),
        "rank": data.get("rank"),
        "popularity": data.get("popularity"),
        "type": media_label(media),
        "year": year,
        "episodes": data.get("num_episodes") or None,
        "status": status.replace("_", " ").title() if status else None,
        "genres": [g["name"] for g in data.get("genres") or []],
        "studios": [s["name"] for s in data.get("studios") or []],
        "synopsis": data.get("synopsis") or None,
    }


def fetch_details(mal_id: int) -> Optional[dict]:
    cache = _load_cache()
    if mal_id in cache:
        return cache[mal_id]

    try:
        from app.clients.mal_api import get_anime_metadata  # lazy: needs MAL_CLIENT_ID
        data = get_anime_metadata(mal_id, fields=DETAIL_FIELDS)
    except RuntimeError:
        return None                                         # MAL_CLIENT_ID missing
    if not data:
        return None

    detail = _map_node(data)
    with _LOCK:
        cache = _load_cache()
        cache[mal_id] = detail
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache))
    return detail
