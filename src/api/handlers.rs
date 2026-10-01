use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use serde_json::json;

use crate::core::dinic::CancellationToken;
use crate::core::errors::{InputError, ServiceError};
use crate::service::contracts::{ComputeRequest, CreateJobRequest, JobResponse};
use crate::service::flow_service::{new_run_id, solve_request};

use super::state::AppState;

const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

#[must_use]
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/max-flow/compute", post(compute))
        .route("/max-flow/jobs", post(create_job))
        .route("/max-flow/jobs/{run_id}", get(get_job))
        .route("/max-flow/jobs/{run_id}/start", post(start_job))
        .route("/max-flow/jobs/{run_id}/cancel", post(cancel_job))
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

async fn compute(body: Bytes) -> Response {
    let run_id = new_run_id();
    let request = match parse_json::<ComputeRequest>(&run_id, body) {
        Ok(request) => request,
        Err(response) => return response,
    };

    let task_run_id = run_id.clone();
    let token = CancellationToken::new();
    let result = tokio::task::spawn_blocking(move || {
        solve_request(request, &token, &task_run_id, &|_event| {})
    })
    .await
    .map_err(|join_error| {
        envelope(
            StatusCode::INTERNAL_SERVER_ERROR,
            Some(&run_id),
            "computation_failed",
            json!({
                "kind": "task_join_failed",
                "message": join_error.to_string(),
            }),
        )
    });

    match result {
        Ok(Ok(response)) => (StatusCode::OK, Json(response)).into_response(),
        Ok(Err(error)) => error.into_response(),
        Err(response) => response,
    }
}

async fn create_job(State(state): State<AppState>, body: Bytes) -> Response {
    let run_id = new_run_id();
    let request = match parse_json::<CreateJobRequest>(&run_id, body) {
        Ok(request) => request,
        Err(response) => return response,
    };

    match state.jobs.create(request) {
        Ok((run_id, response)) => (
            StatusCode::ACCEPTED,
            [(
                axum::http::header::LOCATION,
                format!("/max-flow/jobs/{run_id}"),
            )],
            Json(response),
        )
            .into_response(),
        Err(error) => error.into_response(),
    }
}

async fn get_job(State(state): State<AppState>, Path(run_id): Path<String>) -> Response {
    match state.jobs.get(&run_id) {
        Ok((status, response)) => {
            let code = match status {
                crate::service::job_store::JobStatus::Queued
                | crate::service::job_store::JobStatus::Running
                | crate::service::job_store::JobStatus::CancellationRequested => {
                    StatusCode::ACCEPTED
                }
                crate::service::job_store::JobStatus::Completed
                | crate::service::job_store::JobStatus::Canceled
                | crate::service::job_store::JobStatus::Failed => StatusCode::OK,
            };
            (code, Json(response)).into_response()
        }
        Err(error) => error.into_response(),
    }
}

async fn start_job(State(state): State<AppState>, Path(run_id): Path<String>) -> Response {
    match state.jobs.start(&run_id) {
        Ok(_) => state
            .jobs
            .get(&run_id)
            .map(|(_, response)| (StatusCode::ACCEPTED, Json(response)).into_response())
            .unwrap_or_else(ServiceError::into_response),
        Err(error) => error.into_response(),
    }
}

async fn cancel_job(State(state): State<AppState>, Path(run_id): Path<String>) -> Response {
    match state.jobs.request_cancellation(&run_id) {
        Ok(JobResponse::Canceled(response)) => {
            (StatusCode::OK, Json(JobResponse::Canceled(response))).into_response()
        }
        Ok(response) => (StatusCode::ACCEPTED, Json(response)).into_response(),
        Err(error) => error.into_response(),
    }
}

fn parse_json<T>(run_id: &str, body: Bytes) -> Result<T, Response>
where
    T: serde::de::DeserializeOwned,
{
    if body.len() > MAX_BODY_BYTES {
        return Err(envelope(
            StatusCode::PAYLOAD_TOO_LARGE,
            Some(run_id),
            "resource_exhausted",
            json!({
                "kind": "request_body_too_large",
                "limit_bytes": MAX_BODY_BYTES,
                "actual_bytes": body.len(),
            }),
        ));
    }

    serde_json::from_slice::<T>(&body).map_err(|error| {
        envelope(
            StatusCode::BAD_REQUEST,
            Some(run_id),
            "input_error",
            json!(InputError::MalformedJson {
                message: error.to_string(),
            }),
        )
    })
}

fn envelope<T>(status: StatusCode, run_id: Option<&str>, category: &str, error: T) -> Response
where
    T: Serialize,
{
    let body = Json(json!({
        "run_id": run_id,
        "category": category,
        "error": error,
    }));
    (status, body).into_response()
}
