from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.persistence.repository import SQLiteRepository
from app.services.beta_service import BetaBinomialService


@lru_cache(maxsize=1)
def get_repository() -> SQLiteRepository:
    return SQLiteRepository(get_settings().database_path)


def get_service() -> BetaBinomialService:
    settings = get_settings()
    return BetaBinomialService(get_repository(), settings)

