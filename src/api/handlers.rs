use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use serde::de::DeserializeOwned;
use uuid::Uuid;

use crate::algorithm::{
    delete_edge_once_per_triangle, DecompositionOrder, TrussKernel,
};
use crate::api::dtos::{
    DecomposeRequest, DecomposeResponse, DeleteEdgeRequest, DeleteEdgeResponse,
    WitnessCheckRequest, WitnessCheckResponse,
};
use crate::api::AppState;
use crate::diagnostics::{
    DiagnosticEvent, ErrorCategory, RequestDiagnostics, ServiceError,
};
use crate::graph::normalize_input;
use crate::witness::check_report_witnesses;

pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "ktruss-backend"
    }))
}

pub async fn decompose(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Result<Json<DecomposeResponse>, ServiceError> {
    let request = parse_request::<DecomposeRequest>(headers, &body)?;
    let request_id = request
        .request_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let normalized = normalize_input(&request.graph).map_err(|error| {
        error.with_request_id(request_id.clone())
    })?;

    let include_events = request
        .include_events
        .unwrap_or(state.config.include_events_by_default);
    let order = DecompositionOrder::from(request.queue_order);
    let mut kernel = TrussKernel::new(normalized.graph);
    let report = kernel.decompose_with_order(
        order,
        normalized.input_edge_count,
        normalized.duplicate_edge_count,
        include_events,
    );

    let diagnostics = RequestDiagnostics::with_id(
        request_id,
        true,
        "accepted",
        serde_json::json!({
            "vertices": report.vertex_count,
            "input_edges": report.input_edge_count,
            "normalized_edges": report.normalized_edge_count,
            "duplicates_deduplicated": report.duplicate_edge_count,
            "triangle_count": report.triangle_count,
            "witness_count": report.witnesses.len(),
            "multi_edge_policy": format!("{:?}", request.graph.multi_edge_policy).to_lowercase(),
        }),
    );
    tracing::info!(
        request_id = %diagnostics.request_id,
        decision = "accepted",
        vertex_count = report.vertex_count,
        input_edges = report.input_edge_count,
        normalized_edges = report.normalized_edge_count,
        triangle_count = report.triangle_count,
        "graph decomposition accepted"
    );

    Ok(Json(DecomposeResponse {
        report,
        diagnostics,
    }))
}

pub async fn delete_edge(
    headers: HeaderMap,
    body: String,
) -> Result<Json<DeleteEdgeResponse>, ServiceError> {
    let request = parse_request::<DeleteEdgeRequest>(headers, &body)?;
    let request_id = request
        .request_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let normalized = normalize_input(&request.graph)
        .map_err(|error| error.with_request_id(request_id.clone()))?;
    let mut graph = normalized.graph;
    let report = delete_edge_once_per_triangle(&mut graph, request.edge_id)
        .map_err(|error| error.with_request_id(request_id.clone()))?;

    let diagnostics = RequestDiagnostics::with_id(
        request_id,
        true,
        "accepted",
        serde_json::json!({
            "removed_edge_id": report.removed_edge,
            "affected_triangle_count": report.affected_triangles.len(),
            "triangle_count_after": report.triangle_count_after,
            "normalized_edge_count": graph.edges().len(),
        }),
    );
    tracing::info!(
        request_id = %diagnostics.request_id,
        decision = "accepted",
        removed_edge_id = report.removed_edge,
        affected_triangle_count = report.affected_triangles.len(),
        "edge deletion accepted"
    );

    Ok(Json(DeleteEdgeResponse {
        report,
        diagnostics,
    }))
}

pub async fn check_witness(
    headers: HeaderMap,
    body: String,
) -> Result<Json<WitnessCheckResponse>, ServiceError> {
    let request = parse_request::<WitnessCheckRequest>(headers, &body)?;
    let request_id = request
        .request_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    match check_report_witnesses(&request.report) {
        Ok(()) => {
            let diagnostics = RequestDiagnostics::with_id(
                request_id,
                true,
                "accepted",
                serde_json::json!({
                    "edge_count": request.report.levels.len(),
                    "witness_count": request.report.witnesses.len(),
                }),
            );
            Ok(Json(WitnessCheckResponse {
                valid: true,
                diagnostics,
            }))
        }
        Err(error) => {
            let mut service_error = ServiceError::invalid_witness(
                format!("witness check failed: {}", error.code()),
                serde_json::to_value(&error).unwrap_or_else(|_| {
                    serde_json::json!({"reason": error.code()})
                }),
            );
            service_error.diagnostics.events.push(DiagnosticEvent::new(
                "witness_check",
                "rejected",
                serde_json::json!({"reason": error.code()}),
            ));
            Err(service_error.with_request_id(request_id))
        }
    }
}

fn parse_request<T: DeserializeOwned>(
    headers: HeaderMap,
    body: &str,
) -> Result<T, ServiceError> {
    let header_request_id = headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    serde_json::from_str(body).map_err(|error| {
        let mut service_error = ServiceError::invalid_json(
            error.to_string(),
            serde_json::json!({
                "body_bytes": body.len(),
                "parse_offset": error.line(),
                "parse_line": error.line(),
                "parse_column": error.column(),
            }),
        );
        if let Some(id) = header_request_id {
            service_error = service_error.with_request_id(id);
        }
        service_error
    })
}

// Silence an unused category import if future handlers add validation variants.
#[allow(dead_code)]
fn _category_used(category: ErrorCategory) -> ErrorCategory {
    category
}
