"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api import router
from app.config import settings
from app.logging_config import configure_logging
from app.storage import RequestJournal


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.logger = configure_logging()
    app.state.journal = RequestJournal(settings.database_path)
    app.state.journal.initialize()
    yield


def create_app(database_path: Path | str | None = None) -> FastAPI:
    journal_path = Path(database_path) if database_path is not None else settings.database_path

    @asynccontextmanager
    async def configured_lifespan(app: FastAPI):
        app.state.logger = configure_logging()
        app.state.journal = RequestJournal(journal_path)
        app.state.journal.initialize()
        yield

    app = FastAPI(
        title="Covariance Estimation Backend",
        version=settings.implementation_version,
        description="Shrinkage and robust covariance estimates with independent matrix diagnostics.",
        lifespan=configured_lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
