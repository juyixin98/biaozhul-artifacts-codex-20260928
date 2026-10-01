from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from app.models.contracts import (
    BatchCountsRequest,
    BatchResponse,
    BatchRevisionResponse,
    CreateModelRequest,
    EventResponse,
    ModelResponse,
    PredictionRequest,
    PredictionResponse,
    RevisionRequest,
)
from app.api.dependencies import get_service
from app.services.beta_service import BetaBinomialService


router = APIRouter(prefix="/models", tags=["beta-binomial"])


@router.post("", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
def create_model(
    request: CreateModelRequest,
    service: BetaBinomialService = Depends(get_service),
) -> dict:
    return service.create_model(request.alpha, request.beta, request.reason)


@router.get("/{model_id}", response_model=ModelResponse)
def get_model(
    model_id: str,
    service: BetaBinomialService = Depends(get_service),
) -> dict:
    return service.current_state(model_id)


@router.get("/{model_id}/events", response_model=list[EventResponse])
def list_events(
    model_id: str,
    service: BetaBinomialService = Depends(get_service),
) -> list[dict]:
    return service.list_events(model_id)


@router.get("/{model_id}/batches", response_model=list[BatchResponse])
def list_batches(
    model_id: str,
    service: BetaBinomialService = Depends(get_service),
) -> list[dict]:
    return service.list_batches(model_id)


@router.post("/{model_id}/batches", response_model=BatchResponse, status_code=status.HTTP_201_CREATED)
def add_batch(
    model_id: str,
    request: BatchCountsRequest,
    service: BetaBinomialService = Depends(get_service),
) -> dict:
    return service.add_batch(
        model_id, request.success_count, request.failure_count, request.reason
    )


@router.post(
    "/{model_id}/batches/{batch_id}/revise",
    response_model=BatchRevisionResponse,
    status_code=status.HTTP_201_CREATED,
)
def revise_batch(
    model_id: str,
    batch_id: str,
    request: RevisionRequest,
    service: BetaBinomialService = Depends(get_service),
) -> dict:
    return service.revise_batch(
        model_id,
        batch_id,
        request.success_count,
        request.failure_count,
        request.reason,
    )


@router.post("/{model_id}/batches/{batch_id}/revert", response_model=BatchResponse)
def revert_batch(
    model_id: str,
    batch_id: str,
    reason: str | None = Query(default=None, max_length=500),
    service: BetaBinomialService = Depends(get_service),
) -> dict:
    return service.revert_batch(model_id, batch_id, reason)


@router.post("/{model_id}/batches/{batch_id}/restore", response_model=BatchResponse)
def restore_batch(
    model_id: str,
    batch_id: str,
    reason: str | None = Query(default=None, max_length=500),
    service: BetaBinomialService = Depends(get_service),
) -> dict:
    return service.restore_batch(model_id, batch_id, reason)


@router.get("/{model_id}/posterior")
def posterior(
    model_id: str,
    service: BetaBinomialService = Depends(get_service),
) -> dict:
    return service.posterior_summary(model_id)


@router.post("/{model_id}/predict", response_model=PredictionResponse)
def predict(
    model_id: str,
    request: PredictionRequest,
    service: BetaBinomialService = Depends(get_service),
) -> dict:
    return service.predict(
        model_id,
        request.trials,
        request.include_distribution,
        request.reason,
    )
