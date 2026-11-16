"""GET /api/users/{username}/exists: quick MAL username check for the valid/invalid hint in the UI.

Served by the backend because MAL v2 needs the client id, and Jikan's /users/ endpoint is too
unreliable for a definite answer. Low volume (debounced client-side) + an in-memory cache.
Undetermined (network/5xx) -> 502, and the frontend stays silent (never a false "invalid").
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()

_cache: dict = {}                      # normalized username -> bool (definite answers only)


@router.get("/users/{username}/exists")
def username_exists(username: str) -> dict:
    key = username.strip().lower()
    if key in _cache:
        return {"exists": _cache[key]}
    try:
        from app.clients.mal_api import user_exists  # lazy: needs MAL_CLIENT_ID
        result = user_exists(username.strip())
    except RuntimeError:
        result = None                                # MAL_CLIENT_ID missing
    if result is None:
        raise HTTPException(status_code=502, detail="can't verify username")
    if len(_cache) > 1024:
        _cache.clear()
    _cache[key] = result
    return {"exists": result}
