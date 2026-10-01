"""Version metadata and deterministic run fingerprints."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone

import fastapi
import numpy
import pydantic
import scipy

from .contracts import VersionInfo
from .kernel import KERNEL_VERSION, SCHEMA_VERSION


def version_info() -> VersionInfo:
    return VersionInfo(
        schema_version=SCHEMA_VERSION,
        kernel_version=KERNEL_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        fastapi=fastapi.__version__,
        pydantic=pydantic.__version__,
    )


def deterministic_input_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
