//! Axum routes, request identity, and structured query/update handlers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info, warn};

use crate::operations::Operation;
use crate::service::{ServiceError, UpdateService};

#[derive(Clone)]
struct AppState {
    service: UpdateService,
    next_request_id: Arc<AtomicU64>,
}

#[derive(Debug, Deserialize)]
struct BatchRequest {
    operations: Vec<Operation>,
}

/// Build the complete HTTP application.
pub fn app(service: UpdateService) -> Router {
    let state = AppState {
        service,
        next_request_id: Arc::new(AtomicU64::new(1)),
    };

    Router::new()
        .route("/health", get(health))
        .route("/state", get(get_state))
        .route("/forest", get(forest))
        .route("/updates/batch", post(batch_update))
        .layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": "dynamic-msf"
    }))
}

async fn get_state(State(app): State<AppState>, headers: HeaderMap) -> Json<Value> {
    let request_id = request_id(&app, &headers);
    let snapshot = app.service.snapshot();
    info!(
        request_id = %request_id,
        version = snapshot.version,
        forest_edges = snapshot.forest_edge_ids.len(),
        total_weight = %snapshot.total_weight,
        "served complete graph state"
    );
    Json(json!({
        "request_id": request_id,
        "snapshot": snapshot
    }))
}

async fn forest(State(app): State<AppState>, headers: HeaderMap) -> Json<Value> {
    let request_id = request_id(&app, &headers);
    let snapshot = app.service.snapshot();
    debug!(
        request_id = %request_id,
        version = snapshot.version,
        "served forest query"
    );
    Json(json!({
        "request_id": request_id,
        "version": snapshot.version,
        "total_weight": snapshot.total_weight,
        "forest_edge_ids": snapshot.forest_edge_ids,
        "graph_component_count": snapshot.graph_component_count
    }))
}

async fn batch_update(
    State(app): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let request_id = request_id(&app, &headers);
    let request = match serde_json::from_slice::<BatchRequest>(&body) {
        Ok(request) => request,
        Err(error) => {
            warn!(request_id = %request_id, error = %error, "batch request body could not be decoded");
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "request_id": request_id,
                    "error": {
                        "category": "invalid_json",
                        "message": error.to_string()
                    }
                })),
            )
                .into_response();
        }
    };
    info!(
        request_id = %request_id,
        operation_count = request.operations.len(),
        "accepted batch update request"
    );

    match app.service.apply_checked(&request.operations) {
        Ok(outcome) => {
            info!(
                request_id = %request_id,
                version_from = outcome.version_from,
                version_to = outcome.version_to,
                total_weight = %outcome.state.total_weight,
                forest_edges = outcome.state.forest_edge_ids.len(),
                "published complete graph version"
            );
            (
                StatusCode::OK,
                Json(json!({
                    "request_id": request_id,
                    "result": outcome
                })),
            )
                .into_response()
        }
        Err(service_error) => {
            let status = match &service_error {
                ServiceError::Core(_) => StatusCode::UNPROCESSABLE_ENTITY,
                ServiceError::WitnessRejected(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            let category = service_error.category();
            let witness_failures = service_error.witness_failures();
            error!(
                request_id = %request_id,
                error_category = category,
                failure_count = witness_failures.len(),
                "batch update rejected; published state and version unchanged"
            );

            let mut body = json!({
                "request_id": request_id,
                "error": {
                    "category": category,
                    "message": service_error.to_string(),
                }
            });
            if !witness_failures.is_empty() {
                body["error"]["witness_failures"] = json!(witness_failures);
            }
            if let ServiceError::Core(core_error) = service_error {
                body["error"]["details"] =
                    serde_json::to_value(core_error).expect("core error is serializable");
            }

            (status, Json(body)).into_response()
        }
    }
}

fn request_id(app: &AppState, headers: &HeaderMap) -> String {
    if let Some(value) = headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
    {
        return value.to_string();
    }

    let sequence = app.next_request_id.fetch_add(1, Ordering::Relaxed);
    format!("local-{sequence:012}")
}
