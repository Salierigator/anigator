"""FastAPI dependencies: the service is built once in the lifespan and kept on app.state."""
from __future__ import annotations

from fastapi import Request

from app.services.base import RecommenderService


def get_service(request: Request) -> RecommenderService:
    return request.app.state.service
