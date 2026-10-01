"""Runtime version metadata attached to runs and logs."""

from __future__ import annotations

import platform
import sqlite3

import fastapi
import numpy
import pydantic
import scipy

from . import __version__


def runtime_versions() -> dict[str, str]:
    return {
        "service": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "fastapi": fastapi.__version__,
        "pydantic": pydantic.VERSION,
        "sqlite": sqlite3.sqlite_version,
    }

