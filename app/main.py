"""Uvicorn entrypoint: ``uvicorn app.main:app``."""

from app.api import app

__all__ = ["app"]

