use crate::dto::{status_name, MessageResponse, SolveJobRequest, WitnessRequest, WitnessResponse};
use crate::error::{AppError, ComputationErrorCode, InputErrorCode};
use crate::jobs::{JobState, SharedRegistry};
use crate::witness::verify_flow_certificate;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;

#[derive(Clone)]
pub struct AppState {
    pub jobs: SharedRegistry,
}

#[derive(Serialize)]
struct CreatedJob {
    id: String,
    state: &'static str,
}

#[derive(Serialize)]
struct FullJobResponse {
    #[serde(flatten)]
    base: crate::dto::JobResponse,
    events: Option<Vec<crate::logging::LogEntry>>,
}

pub fn router() -> Router {
    let state = AppState {
        jobs: SharedRegistry::new(),
    };
    Router::new()
        .route("/health", get(health))
        .route("/jobs", post(create_job))
        .route("/jobs/:job_id", get(get_job))
        .route("/jobs/:job_id/cancel", post(cancel_job))
        .route("/witness", post(check_witness))
        .with_state(state)
}

async fn health() -> Json<MessageResponse> {
    Json(MessageResponse {
        message: "maxflow-server is running".to_string(),
    })
}

async fn create_job(
    State(state): State<AppState>,
    request: Result<Json<SolveJobRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<(StatusCode, Json<CreatedJob>), AppError> {
    let Json(request) = request.map_err(json_rejection)?;
    let job_id = state.jobs.submit(request)?;
    Ok((
        StatusCode::CREATED,
        Json(CreatedJob {
            id: job_id,
            state: "running",
        }),
    ))
}

async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<FullJobResponse>, AppError> {
    let guard = state.jobs.inner.lock().map_err(|_| {
        AppError::computation(
            ComputationErrorCode::InternalStatePoisoned,
            "job registry mutex was poisoned",
        )
    })?;
    let entry = guard
        .get(&job_id)
        .ok_or_else(|| AppError::not_found(format!("job {job_id} does not exist")))?;
    let base = entry.response(&job_id);
    let events = entry.outcome.as_ref().map(|outcome| outcome.events.clone());
    Ok(Json(FullJobResponse { base, events }))
}

async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<MessageResponse>, AppError> {
    state.jobs.cancel(&job_id)?;
    Ok(Json(MessageResponse {
        message: format!("cancellation requested for job {job_id}"),
    }))
}

async fn check_witness(
    request: Result<Json<WitnessRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<WitnessResponse>, AppError> {
    let Json(request) = request.map_err(json_rejection)?;
    let (spec, certificate) = request.into_parts()?;
    let require_cut = certificate.cut.is_some();
    let report = verify_flow_certificate(&spec, &certificate, require_cut);
    Ok(Json(WitnessResponse {
        valid: report.valid,
        failures: report.failure_strings(),
        report,
    }))
}

fn json_rejection(error: axum::extract::rejection::JsonRejection) -> AppError {
    AppError::input(
        InputErrorCode::InvalidJson,
        format!("request body is not valid JSON accepted by the contract: {error}"),
    )
}

pub async fn not_found() -> Response {
    AppError::not_found("route does not exist").into_response()
}

#[allow(dead_code)]
fn state_name(state: JobState) -> &'static str {
    match state {
        JobState::Running => "running",
        JobState::Completed => status_name(crate::flow::SolveStatus::Completed),
        JobState::Canceled => status_name(crate::flow::SolveStatus::Canceled),
        JobState::Failed => "failed",
    }
}
