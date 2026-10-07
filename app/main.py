"""Runnable uvicorn entry point: ``python3 -m app.main``."""
from __future__ import annotations

import os

import uvicorn

from .api import create_app
from .storage import ModelStore

DB_PATH = os.environ.get("SGNS_DB_PATH", "data/runs.sqlite")
ARTIFACT_DIR = os.environ.get("SGNS_ARTIFACT_DIR", "data/artifacts")

store = ModelStore(DB_PATH, ARTIFACT_DIR)
app = create_app(store)


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("SGNS_HOST", "127.0.0.1"),
        port=int(os.environ.get("SGNS_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
