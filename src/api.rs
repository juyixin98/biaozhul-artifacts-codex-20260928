use axum::body::Bytes;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::algorithm::{analyze, map_named_order, AnalysisResult};
use crate::errors::{json_parse_error, ApiError};
use crate::graph::{Graph, GraphInput};
use crate::logger::RunContext;
use crate::witness::{find_induced_cycle, peo_violation, PeoViolation};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifyOrderRequest {
    #[serde(flatten)]
    pub graph: GraphInput,
    pub order: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeResponse {
    pub run_id: String,
    #[serde(flatten)]
    pub result: AnalysisResult,
}

#[derive(Debug, Serialize)]
pub struct VerifyOrderResponse {
    pub run_id: String,
    pub valid: bool,
    pub supplied_order: Vec<String>,
    pub supplied_order_ids: Vec<usize>,
    pub supplied_peo_violation: Option<PeoViolation>,
    #[serde(flatten)]
    pub canonical: AnalysisResult,
}

pub fn router() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/graphs/analyze", post(analyze_handler))
        .route("/api/elimination-orders/verify", post(verify_order_handler))
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "service": "chordal-graph-backend"
    }))
}

async fn analyze_handler(bytes: Bytes) -> Result<impl IntoResponse, ApiError> {
    let run = RunContext::new();
    run.event(
        "request",
        "received",
        "POST /api/graphs/analyze",
        json!({"bytes": bytes.len()}),
    );

    let result = parse_and_analyze(&run, &bytes)?;
    run.event(
        "response",
        "success",
        "analysis completed",
        json!({
            "chordal": result.chordal,
            "max_clique_size": result.max_clique_size,
            "maximal_clique_count": result.maximal_cliques.len(),
            "cycle_length": result.chordless_cycle.as_ref().map(|cycle| cycle.length)
        }),
    );

    Ok((
        StatusCode::OK,
        Json(AnalyzeResponse {
            run_id: run.id().to_string(),
            result,
        }),
    ))
}

async fn verify_order_handler(bytes: Bytes) -> Result<impl IntoResponse, ApiError> {
    let run = RunContext::new();
    run.event(
        "request",
        "received",
        "POST /api/elimination-orders/verify",
        json!({"bytes": bytes.len()}),
    );

    let request: VerifyOrderRequest = serde_json::from_slice(&bytes)
        .map_err(json_parse_error)
        .map_err(|error| error.with_run(run.id().to_string()))?;
    run.event(
        "input",
        "parsed",
        "request parsed",
        json!({
            "vertex_count": request.graph.vertices.len(),
            "raw_edge_count": request.graph.edges.len(),
            "order_length": request.order.len()
        }),
    );

    let graph = Graph::from_input(&request.graph)
        .map_err(ApiError::input)
        .map_err(|error| error.with_run(run.id().to_string()))?;
    run.event(
        "graph",
        "normalized",
        "graph converted to simple undirected graph",
        json!({
            "vertices": graph.vertex_count(),
            "normalized_edges": graph.edge_count(),
            "duplicate_edges": graph.duplicate_edge_count(),
            "components": graph.components().len()
        }),
    );

    let order_ids = map_named_order(&graph, &request.order)
        .map_err(ApiError::input)
        .map_err(|error| error.with_run(run.id().to_string()))?;

    let canonical = checked_analysis(&run, &graph)?;
    if !canonical.chordal {
        let cycle = canonical
            .chordless_cycle
            .clone()
            .or_else(|| find_induced_cycle(&graph));
        run.event(
            "verification",
            "state_conflict",
            "graph is not chordal; no supplied order can be a perfect elimination order",
            json!({"cycle_length": cycle.as_ref().map(|witness| witness.length)}),
        );
        return Err(ApiError::state_conflict(
            "non_chordal_graph",
            "the graph is not chordal; no supplied order can be a perfect elimination order"
                .to_string(),
            json!({
                "supplied_order": request.order,
                "chordless_cycle": cycle,
                "canonical_elimination_order": canonical.elimination_order
            }),
        )
        .with_run(run.id().to_string()));
    }

    match peo_violation(&graph, &order_ids) {
        None => {
            run.event(
                "verification",
                "success",
                "supplied elimination order passed later-neighbor clique checks",
                json!({"order": request.order}),
            );
            Ok((
                StatusCode::OK,
                Json(VerifyOrderResponse {
                    run_id: run.id().to_string(),
                    valid: true,
                    supplied_order: request.order,
                    supplied_order_ids: order_ids,
                    supplied_peo_violation: None,
                    canonical,
                }),
            ))
        }
        Some(violation) => {
            run.event(
                "verification",
                "state_conflict",
                "supplied order is not a perfect elimination order",
                json!({
                    "vertex": violation.vertex,
                    "missing_edge": violation.missing_edge
                }),
            );
            Err(ApiError::state_conflict(
                "supplied_order_not_peo",
                "supplied order fails the later-neighbor clique condition".to_string(),
                json!({
                    "supplied_order": request.order,
                    "supplied_peo_violation": violation,
                    "canonical_elimination_order": canonical.elimination_order
                }),
            )
            .with_run(run.id().to_string()))
        }
    }
}

fn parse_and_analyze(run: &RunContext, bytes: &[u8]) -> Result<AnalysisResult, ApiError> {
    let input: GraphInput = serde_json::from_slice(bytes)
        .map_err(json_parse_error)
        .map_err(|error| error.with_run(run.id().to_string()))?;
    run.event(
        "input",
        "parsed",
        "graph input parsed",
        json!({
            "vertex_count": input.vertices.len(),
            "raw_edge_count": input.edges.len()
        }),
    );

    let graph = Graph::from_input(&input)
        .map_err(ApiError::input)
        .map_err(|error| error.with_run(run.id().to_string()))?;
    run.event(
        "graph",
        "normalized",
        "graph converted to simple undirected graph",
        json!({
            "vertices": graph.vertex_count(),
            "normalized_edges": graph.edge_count(),
            "duplicate_edges": graph.duplicate_edge_count(),
            "components": graph.components().len()
        }),
    );

    checked_analysis(run, &graph)
}

fn checked_analysis(run: &RunContext, graph: &Graph) -> Result<AnalysisResult, ApiError> {
    let result = analyze(graph);
    run.event(
        "algorithm",
        "computed",
        "maximum cardinality search and witness analysis completed",
        json!({
            "chordal": result.chordal,
            "search_order": result.mcs_search_order,
            "elimination_order": result.elimination_order,
            "peo_violation": result.peo_violation,
            "cycle_length": result.chordless_cycle.as_ref().map(|cycle| cycle.length),
            "max_clique_size": result.max_clique_size,
            "maximal_cliques": result.maximal_cliques
        }),
    );

    if let Some(checks) = &result.checks {
        run.event(
            "witness_checks",
            if [
                checks.peo.passed,
                checks.maximal_cliques.passed,
                checks.clique_cover.passed,
                checks.running_intersection.passed,
            ]
            .iter()
            .all(|passed| *passed)
            {
                "passed"
            } else {
                "failed"
            },
            "independent checks evaluated",
            serde_json::to_value(checks).unwrap_or_default(),
        );

        if !checks.peo.passed
            || !checks.maximal_cliques.passed
            || !checks.clique_cover.passed
            || !checks.running_intersection.passed
        {
            return Err(ApiError::computation(
                "internal algorithm result failed an independently evaluated invariant".to_string(),
                serde_json::to_value(checks).unwrap_or_default(),
            )
            .with_run(run.id().to_string()));
        }
    } else if result.chordless_cycle.is_none() {
        return Err(ApiError::computation(
            "algorithm reported a non-chordal graph but did not produce a chordless cycle"
                .to_string(),
            json!({"elimination_order": result.elimination_order}),
        )
        .with_run(run.id().to_string()));
    }

    Ok(result)
}
