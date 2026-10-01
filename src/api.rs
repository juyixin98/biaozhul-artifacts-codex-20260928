//! Axum backend queries and response projections.

use crate::config::Settings;
use crate::graph::{Diagnostic, GraphError, GraphInput, NormalizationReport, SimpleGraph};
use crate::kernel::{Decomposition, EdgeDecomposition, PeelingOrder, Witness, decompose};
use crate::witness::verify_decomposition;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct DecomposeQuery {
    #[serde(default)]
    pub edge_id: Option<String>,
    #[serde(default)]
    pub peeling_order: Option<PeelingOrderParam>,
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PeelingOrderParam {
    #[default]
    Declaration,
    ReverseDeclaration,
    EndpointLex,
}

impl From<PeelingOrderParam> for PeelingOrder {
    fn from(value: PeelingOrderParam) -> Self {
        match value {
            PeelingOrderParam::Declaration => PeelingOrder::Declaration,
            PeelingOrderParam::ReverseDeclaration => PeelingOrder::ReverseDeclaration,
            PeelingOrderParam::EndpointLex => PeelingOrder::EndpointLex,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DecomposeResponse {
    pub request_id: String,
    pub disposition: String,
    pub graph_name: Option<String>,
    pub triangle_count: usize,
    pub normalization: NormalizationReport,
    pub peeling_order: PeelingOrder,
    pub verification_accepted: bool,
    pub edges: Vec<EdgeDecomposition>,
    pub witnesses: Vec<Witness>,
    pub counters: crate::kernel::KernelCounters,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub request_id: String,
    pub disposition: String,
    pub code: String,
    pub message: String,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug)]
pub struct ApiError {
    request_id: String,
    status: StatusCode,
    code: String,
    message: String,
    diagnostics: Vec<Diagnostic>,
}

impl ApiError {
    fn bad_request(request_id: String, message: String) -> Self {
        Self {
            request_id,
            status: StatusCode::BAD_REQUEST,
            code: "MALFORMED_REQUEST".into(),
            message,
            diagnostics: Vec::new(),
        }
    }

    fn from_graph_error(request_id: String, error: GraphError, redact: bool) -> Self {
        let diagnostics = if redact {
            error
                .diagnostics
                .into_iter()
                .map(redact_diagnostic)
                .collect()
        } else {
            error.diagnostics
        };
        Self {
            request_id,
            status: StatusCode::from_u16(error.status).unwrap_or(StatusCode::UNPROCESSABLE_ENTITY),
            code: error.code,
            message: error.message,
            diagnostics,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        warn!(
            request_id = %self.request_id,
            code = %self.code,
            status = %self.status.as_u16(),
            diagnostic_count = self.diagnostics.len(),
            "request rejected"
        );
        (
            self.status,
            Json(ErrorResponse {
                request_id: self.request_id,
                disposition: "rejected".into(),
                code: self.code,
                message: self.message,
                diagnostics: self.diagnostics,
            }),
        )
            .into_response()
    }
}

pub fn router(settings: Settings) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/v1/graphs/analyze", post(analyze_graph))
        .with_state(AppState {
            settings: Arc::new(settings),
        })
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        service: "triangular-support-ktruss".into(),
    })
}

async fn analyze_graph(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Query(query): axum::extract::Query<DecomposeQuery>,
    raw_body: axum::body::Bytes,
) -> Result<Json<DecomposeResponse>, ApiError> {
    let request_id = request_id(&headers);
    let order = query.peeling_order.unwrap_or_default().into();

    let input: GraphInput = serde_json::from_slice(&raw_body).map_err(|error| {
        ApiError::bad_request(
            request_id.clone(),
            format!("request body is not valid GraphInput JSON: {error}"),
        )
    })?;

    if input.nodes.len() > state.settings.max_nodes {
        return Err(limit_error(
            request_id,
            "NODE_LIMIT_EXCEEDED",
            "declared node count exceeds configured local limit",
            "node_count",
            input.nodes.len(),
            state.settings.max_nodes,
        ));
    }
    if input.edges.len() > state.settings.max_edges {
        return Err(limit_error(
            request_id,
            "EDGE_LIMIT_EXCEEDED",
            "declared edge count exceeds configured local limit",
            "edge_count",
            input.edges.len(),
            state.settings.max_edges,
        ));
    }

    let graph_name = input.name.clone();
    let (graph, normalization) = SimpleGraph::from_input(input).map_err(|error| {
        ApiError::from_graph_error(request_id.clone(), error, state.settings.redact_diagnostics)
    })?;

    info!(
        request_id = %request_id,
        nodes = graph.node_count(),
        edges = graph.edge_count(),
        duplicate_declarations = normalization.removed_duplicate_edge_declarations,
        "normalized graph accepted"
    );

    let decomposition = decompose(&graph, order).map_err(|error| ApiError {
        request_id: request_id.clone(),
        status: StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        code: error.code,
        message: error.message,
        diagnostics: Vec::new(),
    })?;
    let verification = verify_decomposition(&graph, &decomposition);

    if !verification.accepted {
        warn!(
            request_id = %request_id,
            diagnostic_count = verification.diagnostics.len(),
            "decomposition witness verification failed"
        );
        return Err(ApiError {
            request_id,
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "WITNESS_VERIFICATION_FAILED".into(),
            message: "server produced a decomposition that failed local witness verification"
                .into(),
            diagnostics: verification.diagnostics,
        });
    }

    let response = project_response(
        request_id,
        graph_name,
        normalization,
        order,
        decomposition,
        query.edge_id,
    )?;

    info!(
        request_id = %response.request_id,
        triangles = response.triangle_count,
        edge_results = response.edges.len(),
        witnesses = response.witnesses.len(),
        updates = response.counters.affected_triangle_updates,
        "decomposition accepted"
    );
    Ok(Json(response))
}

fn project_response(
    request_id: String,
    graph_name: Option<String>,
    normalization: NormalizationReport,
    order: PeelingOrder,
    decomposition: Decomposition,
    edge_id: Option<String>,
) -> Result<DecomposeResponse, ApiError> {
    let triangle_count = decomposition.counters.triangle_count;
    let verification_accepted = true;
    let diagnostics = Vec::new();

    let (edges, witnesses) = match edge_id {
        None => (decomposition.edges.clone(), decomposition.witnesses.clone()),
        Some(edge_id) => {
            let edge = decomposition
                .edges
                .iter()
                .find(|edge| edge.edge_id == edge_id)
                .ok_or_else(|| ApiError {
                    request_id: request_id.clone(),
                    status: StatusCode::NOT_FOUND,
                    code: "EDGE_NOT_FOUND".into(),
                    message: "requested normalized edge id is absent".into(),
                    diagnostics: vec![
                        Diagnostic::new(
                            "EDGE_NOT_FOUND",
                            "edge_id query parameter did not match any normalized edge",
                        )
                        .with("redacted_edge_id", redact_string(&edge_id)),
                    ],
                })?
                .clone();
            let witness = decomposition
                .witnesses
                .iter()
                .find(|witness| witness.witness_id == edge.witness_id)
                .cloned()
                .expect("verified decomposition must contain edge witness");
            (vec![edge], vec![witness])
        }
    };

    Ok(DecomposeResponse {
        request_id,
        disposition: "accepted".into(),
        graph_name,
        triangle_count,
        normalization,
        peeling_order: order,
        verification_accepted,
        edges,
        witnesses,
        counters: decomposition.counters,
        diagnostics,
    })
}

fn request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 128
                && value
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
        })
        .map(str::to_string)
        .unwrap_or_else(|| format!("req-{:016x}", pseudo_random()))
}

fn pseudo_random() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ std::process::id() as u64 ^ 0x9e37_79b9_7f4a_7c15
}

fn limit_error(
    request_id: String,
    code: &str,
    message: &str,
    field: &str,
    actual: usize,
    limit: usize,
) -> ApiError {
    ApiError {
        request_id,
        status: StatusCode::PAYLOAD_TOO_LARGE,
        code: code.into(),
        message: message.into(),
        diagnostics: vec![
            Diagnostic::new(code, message)
                .with(field, actual.to_string())
                .with("configured_limit", limit.to_string()),
        ],
    }
}

fn redact_diagnostic(diagnostic: Diagnostic) -> Diagnostic {
    Diagnostic {
        code: diagnostic.code,
        message: diagnostic.message,
        state: diagnostic
            .state
            .into_iter()
            .map(|(key, value)| {
                let redacted_value = if key.contains("edge_id") {
                    value
                } else {
                    redact_string(&value)
                };
                (key, redacted_value)
            })
            .collect(),
    }
}

fn redact_string(value: &str) -> String {
    crate::graph::redact_public(value)
}
