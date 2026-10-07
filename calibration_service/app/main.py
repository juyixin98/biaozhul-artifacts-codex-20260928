"""FastAPI entrypoint: /fit, /predict, /evaluate, /models, /health."""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .artifact import ArtifactStore
from .calibrator import IsotonicCalibrator
from .config import LIB_VERSION, settings
from .evaluation import evaluate
from .logging_setup import configure_logging, get_logger, new_request_id
from .schemas import EvaluateRequest, FitRequest, PredictRequest
from .validation import ValidationError

configure_logging()
log = get_logger("api")

app = FastAPI(title="isotonic-calibration-service", version=LIB_VERSION)
store = ArtifactStore(settings.db_path)


def _error(request_id: str, status: int, category: str, message: str, detail=None):
    log.warning("request failed category=%s reason=%s", category, message)
    return JSONResponse(
        status_code=status,
        content={
            "request_id": request_id,
            "error": message,
            "category": category,
            "detail": detail,
        },
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = new_request_id()
    log.info("start %s %s", request.method, request.url.path)
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    log.info("done %s %s status=%d", request.method, request.url.path, response.status_code)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "lib_version": LIB_VERSION}


@app.post("/fit")
def fit(req: FitRequest):
    rid = new_request_id()
    ds = req.dataset
    weights = ds.weights if ds.weights is not None else [1.0] * len(ds.scores)
    score_field = req.score_field or settings.score_field
    log.info(
        "fit model_id=%s rows=%d score_field=%s", req.model_id, len(ds.scores), score_field
    )
    try:
        calibrator = IsotonicCalibrator.fit(
            np.asarray(ds.scores), np.asarray(ds.labels), np.asarray(weights),
            score_field=score_field,
        )
    except ValidationError as e:
        return _error(rid, 422, "validation_error", str(e), e.row_indices)
    summary = calibrator.validation_summary
    log.info(
        "fit done model_id=%s blocks=%d used=%d dropped_zero_weight=%d merged_dup=%d",
        req.model_id, calibrator.probabilities.size, summary.n_used,
        summary.n_zero_weight_dropped, summary.n_duplicate_scores_merged,
    )
    meta = store.save(req.model_id, calibrator, LIB_VERSION)
    log.info("artifact saved model_id=%s version=%d", req.model_id, meta["version"])
    return {
        "request_id": rid,
        "model_id": req.model_id,
        "version": meta["version"],
        "lib_version": LIB_VERSION,
        "score_field": score_field,
        "extrapolation": calibrator.extrapolation,
        "n_blocks": int(calibrator.probabilities.size),
        "monotone_non_decreasing": calibrator.is_monotone(),
        "validation": {
            "n_input": summary.n_input,
            "n_used": summary.n_used,
            "n_zero_weight_dropped": summary.n_zero_weight_dropped,
            "n_duplicate_scores_merged": summary.n_duplicate_scores_merged,
            "warnings": summary.warnings,
        },
        "mapping": calibrator.to_dict(),
    }


def _load_or_error(rid: str, model_id: str, version: int | None):
    try:
        return store.load(model_id, version), None
    except KeyError as e:
        return None, _error(rid, 404, "not_found", str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    rid = new_request_id()
    loaded, err = _load_or_error(rid, req.model_id, req.version)
    if err:
        return err
    calibrator, meta = loaded
    scores = np.asarray(req.scores, dtype=float)
    if not np.isfinite(scores).all():
        return _error(rid, 422, "validation_error", "scores must be finite")
    probs = calibrator.predict(scores)
    log.info(
        "predict model_id=%s version=%d n=%d score_field=%s",
        req.model_id, meta["version"], len(req.scores), meta["score_field"],
    )
    return {
        "request_id": rid,
        "model_id": req.model_id,
        "version": meta["version"],
        "score_field": meta["score_field"],
        "extrapolation": calibrator.extrapolation,
        "probabilities": probs.tolist(),
    }


@app.post("/evaluate")
def evaluate_endpoint(req: EvaluateRequest):
    rid = new_request_id()
    loaded, err = _load_or_error(rid, req.model_id, req.version)
    if err:
        return err
    calibrator, meta = loaded
    ds = req.dataset
    weights = ds.weights if ds.weights is not None else [1.0] * len(ds.scores)
    try:
        metrics = evaluate(
            calibrator,
            np.asarray(ds.scores),
            np.asarray(ds.labels),
            np.asarray(weights),
            n_bins=settings.ece_bins,
        )
    except ValidationError as e:
        return _error(rid, 422, "validation_error", str(e), e.row_indices)
    log.info(
        "evaluate model_id=%s version=%d n_eval=%d log_loss=%.6f brier=%.6f ece=%.6f",
        req.model_id, meta["version"], metrics["n_eval_rows"],
        metrics["log_loss"], metrics["brier_score"], metrics["ece"],
    )
    return {
        "request_id": rid,
        "model_id": req.model_id,
        "version": meta["version"],
        "metrics": metrics,
    }


@app.get("/models")
def models():
    return {"lib_version": LIB_VERSION, "artifacts": store.list_models()}
