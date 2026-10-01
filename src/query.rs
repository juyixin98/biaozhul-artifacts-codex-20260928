//! Axum HTTP query layer.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use serde_json::json;
use uuid::Uuid;

use crate::algorithm::{solve_dag_path_cover, SolveError};
use crate::config::Config;
use crate::model::GraphInput;

#[derive(Debug, Clone)]
pub struct AppState {
    pub config: Config,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    status: &'static str,
    code: String,
    message: String,
    run_id: String,
    version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<serde_json::Value>,
}

pub fn router(config: Config) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/v1/path-cover/minimum", post(minimum_path_cover))
        .with_state(AppState { config })
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "service": "dag-path-cover",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

async fn minimum_path_cover(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<GraphInput>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let run_id = run_id_from_headers(&headers);
    tracing::info!(run_id = %run_id, "received minimum path cover query");

    let Json(input) = match body {
        Ok(input) => input,
        Err(rejection) => {
            let raw_status = rejection.status();
            let status = if raw_status.is_client_error() {
                StatusCode::BAD_REQUEST
            } else {
                raw_status
            };
            tracing::error!(run_id = %run_id, error = %rejection, "malformed JSON request");
            return build_response(
                status,
                ErrorBody {
                    status: "error",
                    code: "invalid_json".to_string(),
                    message: rejection.body_text(),
                    run_id,
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    details: None,
                },
            );
        }
    };

    let max_vertices = state.config.max_vertices;
    let worker_run_id = run_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        solve_dag_path_cover(&worker_run_id, &input, max_vertices)
    })
    .await;

    match result {
        Ok(Ok(response)) => (StatusCode::OK, Json(response)).into_response(),
        Ok(Err(error)) => error_response(&run_id, error),
        Err(join_error) => build_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorBody {
                status: "error",
                code: "compute_task_failed".to_string(),
                message: format!("algorithm worker failed: {join_error}"),
                run_id,
                version: env!("CARGO_PKG_VERSION").to_string(),
                details: None,
            },
        ),
    }
}

fn run_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("x-run-id")
        .or_else(|| headers.get("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty() && value.len() <= 128 && value.is_ascii())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

fn error_response(run_id: &str, error: SolveError) -> Response {
    let status =
        StatusCode::from_u16(error.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let code = error.code().to_string();
    let message = error.message();
    let details = serde_json::to_value(&error).ok();

    build_response(
        status,
        ErrorBody {
            status: "error",
            code,
            message,
            run_id: run_id.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            details,
        },
    )
}

fn build_response(status: StatusCode, body: ErrorBody) -> Response {
    (status, Json(body)).into_response()
}
