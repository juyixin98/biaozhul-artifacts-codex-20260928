"""Wire envelope shared by the local SUT implementations.

This module deliberately contains *no* state-machine logic: the reference and
buggy services own their transition rules independently.
"""
from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def ok(state: str, message: str = "", *, exists: bool = True, code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"ok": True, "state": state, "exists": exists, "message": message},
    )


def reject(
    code: int,
    error: str,
    *,
    state: str | None = None,
    exists: bool = True,
) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"ok": False, "state": state, "exists": exists, "error": error},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Normalise FastAPI's 422 payload into the same {ok:false} envelope.
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "state": None,
            "exists": False,
            "error": "invalid request body",
            "detail": exc.errors(),
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "state": None,
            "exists": False,
            "error": f"internal error: {type(exc).__name__}",
        },
    )
