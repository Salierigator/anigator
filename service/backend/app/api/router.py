"""Collects every route module. A new feature = a new module in routes/, included here."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import anime, health, posters, recommend, users

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(recommend.router, tags=["recommend"])
api_router.include_router(posters.router, tags=["posters"])
api_router.include_router(anime.router, tags=["anime"])
api_router.include_router(users.router, tags=["users"])
