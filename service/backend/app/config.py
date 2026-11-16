"""Central settings (pydantic-settings), read from the environment and service/.env.

Mock mode (`MOCK_MODE=1`) is the default, so the frontend can be developed without the model;
`MOCK_MODE=0` makes the lifespan load the real Recommender.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND = Path(__file__).resolve().parents[1]             # service/backend/
ROOT = Path(__file__).resolve().parents[3]                # repo root
sys.path.insert(0, str(ROOT))                             # the one place that makes `model` importable


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / "service" / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    mock_mode: bool = True
    cors_origins: List[str] = ["http://localhost:5173"]   # Vite dev server
    fixtures_dir: Path = BACKEND / "fixtures"


@lru_cache
def get_settings() -> Settings:
    return Settings()
