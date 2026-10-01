from __future__ import annotations

from functools import lru_cache

from .config import Settings, get_settings
from .logging_config import configure_logging
from .repository import DiagnosticRepository


@lru_cache(maxsize=1)
def cached_settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def cached_repository() -> DiagnosticRepository:
    return DiagnosticRepository(cached_settings())


def get_logger():
    return configure_logging(cached_settings())


def get_repository() -> DiagnosticRepository:
    settings = cached_settings()
    if settings.persist_results:
        return cached_repository()
    return DiagnosticRepository(settings)
