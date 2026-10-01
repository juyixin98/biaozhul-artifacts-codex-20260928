"""HTTP routes for covariance estimation and request inspection."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.errors import CovarianceError, FailureRecord
from app.models import EstimationRequest
from app.service import estimate, resolve_request_id
from app.storage import RequestJournal
from app.versioning import error_envelope

router = APIRouter(prefix="/api/v1", tags=["covariance"])


def get_journal(request: Request) -> RequestJournal:
    return request.app.state.journal


@router.get("/health")
def health(journal: RequestJournal = Depends(get_journal)) -> dict[str, str]:
    journal.initialize()
    return {"status": "ok", "storage": str(journal.database_path)}


@router.post("/covariance/estimate")
def estimate_covariance(
    request: EstimationRequest,
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    journal: RequestJournal = Depends(get_journal),
):
    try:
        request_id = resolve_request_id(request, x_request_id)
        response = estimate(
            request,
            journal=journal,
            logger=request.app.state.logger,
            request_id=request_id,
        )
    except CovarianceError as error:
        failure = FailureRecord.from_error(error)
        request_id = x_request_id or request.request_id or "unidentified"
        payload = request.model_dump(mode="json")
        response = error_envelope(request_id=request_id, failure=failure, payload=payload)
        journal.record(
            request_id=request_id,
            created_at=response["created_at"],
            status="failed",
            methods=list(map(str, request.methods)),
            n_samples=len(request.rows),
            n_features=len(request.rows[0]) if request.rows else None,
            payload=payload,
            response=None,
            failure=failure.to_dict(),
        )
        status_code = 409 if error.stage == "request_identity" else 400
        raise HTTPException(status_code=status_code, detail=response) from error

    if response["status"] == "failed":
        stage = response["failures"][0]["stage"] if response["failures"] else "unknown"
        status_code = 500 if "validation" in stage else 422
        raise HTTPException(status_code=status_code, detail=response)
    return response


@router.get("/covariance/requests/{request_id}")
def fetch_request(request_id: str, journal: RequestJournal = Depends(get_journal)) -> dict:
    item = journal.fetch(request_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"request_id": request_id, "message": "request not found"})
    return item
