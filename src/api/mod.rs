use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::dto::{AnalyzeRequest, VerifyOrderRequest};
use crate::error::{ApiError, ApiResult, ErrorEnvelope, ErrorKind};
use crate::graph::Graph;
use crate::kernel::{analyze_chordality, verify_custom_order, DEFAULT_CYCLE_SEARCH_BUDGET};
use crate::logging::{current_timestamp, error_event, new_run_id, RunLogger};

mod response;

#[derive(Clone)]
pub struct AppState {
    logger: Arc<RunLogger>,
}

impl AppState {
    pub fn new(log_path: &str) -> Self {
        Self {
            logger: Arc::new(RunLogger::new(log_path)),
        }
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/chordality/analyze", post(analyze))
        .route("/api/chordality/verify-order", post(verify_order))
        .fallback(not_found)
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": "chordal-api",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn analyze(State(state): State<AppState>, body: axum::body::Bytes) -> Response {
    let run_id = extract_run_id(&body);
    match analyze_inner(&body) {
        Ok(mut value) => {
            value["run_id"] = json!(run_id);
            log_ok(&state, &run_id, "/api/chordality/analyze");
            (StatusCode::OK, Json(value)).into_response()
        }
        Err(error) => error_response(error, run_id, state, "/api/chordality/analyze"),
    }
}

async fn verify_order(State(state): State<AppState>, body: axum::body::Bytes) -> Response {
    let run_id = extract_run_id(&body);
    match verify_order_inner(&body) {
        Ok(mut value) => {
            value["run_id"] = json!(run_id);
            log_ok(&state, &run_id, "/api/chordality/verify-order");
            (StatusCode::OK, Json(value)).into_response()
        }
        Err(error) => error_response(error, run_id, state, "/api/chordality/verify-order"),
    }
}

async fn not_found() -> ApiError {
    ApiError::NotFound("route_not_found", "unknown API route".to_owned())
}

fn analyze_inner(body: &[u8]) -> ApiResult<Value> {
    let request: AnalyzeRequest = parse_json(body)?;
    let run_id = normalize_run_id(request.run_id)?;
    let (vertices, edges) = request.graph.into_parts();
    let graph = Graph::from_parts(vertices, edges).map_err(ApiError::from_build)?;
    let run =
        analyze_chordality(&graph, DEFAULT_CYCLE_SEARCH_BUDGET).map_err(ApiError::from_analysis)?;

    Ok(json!({
        "run_id": run_id,
        "result": response::analysis_response(&graph, &run),
    }))
}

fn verify_order_inner(body: &[u8]) -> ApiResult<Value> {
    let request: VerifyOrderRequest = parse_json(body)?;
    let run_id = normalize_run_id(request.run_id)?;
    let (vertices, edges) = request.graph.into_parts();
    let graph = Graph::from_parts(vertices, edges).map_err(ApiError::from_build)?;

    let mut order = Vec::with_capacity(request.order.len());
    for input in request.order {
        let label = input.into_label();
        let id = graph.vertex_id(&label).ok_or_else(|| {
            ApiError::Input(
                "unknown_order_vertex",
                format!("order references unknown vertex {label}"),
            )
        })?;
        order.push(id);
    }

    let verification = verify_custom_order(&graph, &order, DEFAULT_CYCLE_SEARCH_BUDGET)
        .map_err(ApiError::from_analysis)?;

    Ok(json!({
        "run_id": run_id,
        "result": response::order_verification_response(&graph, &order, &verification),
    }))
}

fn parse_json<T: serde::de::DeserializeOwned>(body: &[u8]) -> ApiResult<T> {
    serde_json::from_slice(body).map_err(|error| {
        ApiError::Input(
            "invalid_json",
            format!("request body is not valid JSON: {error}"),
        )
    })
}

fn extract_run_id(body: &[u8]) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("run_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .filter(|value| valid_run_id(value))
        .unwrap_or_else(new_run_id)
}

fn normalize_run_id(run_id: Option<String>) -> ApiResult<String> {
    match run_id {
        Some(run_id) if valid_run_id(&run_id) => Ok(run_id),
        Some(_) => Err(ApiError::Input(
            "invalid_run_id",
            "run_id must contain 1-64 ASCII letters, digits, '_', '-', '.' or ':'".to_owned(),
        )),
        None => Ok(new_run_id()),
    }
}

fn valid_run_id(run_id: &str) -> bool {
    let length = run_id.chars().count();
    (1..=64).contains(&length)
        && run_id.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | ':')
        })
}

fn log_ok(state: &AppState, run_id: &str, endpoint: &str) {
    let _ = state.logger.append(json!({
        "timestamp": current_timestamp(),
        "run_id": run_id,
        "endpoint": endpoint,
        "status": "ok",
    }));
}

fn error_response(error: ApiError, run_id: String, state: AppState, endpoint: &str) -> Response {
    let kind = error.kind();
    let code = error.code();
    let message = error.message().to_owned();
    let status = match kind {
        ErrorKind::InputError => StatusCode::BAD_REQUEST,
        ErrorKind::StateConflict => StatusCode::CONFLICT,
        ErrorKind::ResourceExhausted => StatusCode::PAYLOAD_TOO_LARGE,
        ErrorKind::ComputationFailure => StatusCode::UNPROCESSABLE_ENTITY,
        ErrorKind::NotFound => StatusCode::NOT_FOUND,
    };

    let _ = state.logger.append(error_event(
        &run_id,
        endpoint,
        &format!("{kind:?}"),
        code,
        &message,
    ));

    (
        status,
        Json(json!({
            "error": ErrorEnvelope {
                kind,
                code,
                message,
                run_id: Some(run_id),
            }
        })),
    )
        .into_response()
}
