use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;

use crate::config::Config;
use crate::dto::{
    DiagnosticsDto, ErrorBody, ErrorResponse, GraphRequest, HealthResponse, LabeledEdgeDto,
    PathCoverResult, StepDto, SuccessResponse,
};
use crate::fixtures;
use crate::model::LabeledGraph;
use crate::service::{analyze_named_graph, AnalysisFailure, AnalysisSuccess};
use crate::{APP_NAME, APP_VERSION};

#[derive(Clone)]
pub struct AppState {
    pub config: Config,
}

pub fn router(config: Config) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/fixtures", get(list_fixtures))
        .route("/api/fixtures/:name", get(get_fixture))
        .route("/api/path-cover", post(path_cover))
        .fallback(not_found)
        .with_state(AppState { config })
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        service: APP_NAME.to_string(),
        version: APP_VERSION.to_string(),
        bind_address: state.config.bind_address.to_string(),
        max_vertices: state.config.max_vertices,
        max_edges: state.config.max_edges,
    })
}

async fn list_fixtures() -> Json<serde_json::Value> {
    Json(json!({
        "ok": true,
        "fixtures": fixtures::summaries()
    }))
}

async fn get_fixture(Path(name): Path<String>) -> Response {
    match fixtures::document(&name) {
        Some(document) => Json(json!({ "ok": true, "fixture": document })).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "ok": false,
                "error": {
                    "category": "FIXTURE_NOT_FOUND",
                    "message": "no embedded fixture has that name"
                }
            })),
        )
            .into_response(),
    }
}

async fn path_cover(State(state): State<AppState>, headers: HeaderMap, body: Body) -> Response {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if !content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .eq_ignore_ascii_case("application/json")
    {
        return simple_error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "UNSUPPORTED_MEDIA_TYPE",
            "set Content-Type to application/json",
        );
    }

    let bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return simple_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "REQUEST_BODY_TOO_LARGE",
                "JSON request body exceeds 1 MiB",
            );
        }
    };

    let request: GraphRequest = match serde_json::from_slice(&bytes) {
        Ok(request) => request,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    ok: false,
                    error: ErrorBody {
                        category: "INVALID_JSON".to_string(),
                        message: "request body must match the documented JSON schema".to_string(),
                        case_id: None,
                        run_id: "run-unparsed".to_string(),
                        input_id: "in-unparsed".to_string(),
                        details: json!({ "serde_message": error.to_string() }),
                        steps: Vec::new(),
                    },
                }),
            )
                .into_response();
        }
    };

    let case_id = request.case_id;
    let graph: LabeledGraph = request.graph.into();
    match analyze_named_graph(
        case_id,
        graph,
        state.config.max_vertices,
        state.config.max_edges,
    ) {
        Ok(success) => (
            StatusCode::OK,
            Json(SuccessResponse {
                ok: true,
                result: success_response(success),
            }),
        )
            .into_response(),
        Err(failure) => failure.into_response(),
    }
}

fn success_response(success: AnalysisSuccess) -> PathCoverResult {
    let label = |id: usize| success.labels[id].clone();
    let paths = success
        .cover
        .paths
        .iter()
        .map(|path| path.iter().map(|id| label(*id)).collect())
        .collect();
    let matching_pairs = success
        .cover
        .matching
        .pairs
        .iter()
        .map(|(from, to)| LabeledEdgeDto {
            from: label(*from),
            to: label(*to),
        })
        .collect();

    PathCoverResult {
        case_id: success.context.case_id,
        run_id: success.context.run_id,
        input_id: success.context.input_id,
        version: success.context.version,
        vertex_count: success.graph.vertex_count,
        edge_count: success.graph.edges.len(),
        minimum_path_count: success.cover.minimum_path_count,
        matching_size: success.cover.matching.size,
        paths,
        matching_pairs,
        sources: success.graph.sources().into_iter().map(label).collect(),
        sinks: success.graph.sinks().into_iter().map(label).collect(),
        diagnostics: DiagnosticsDto {
            valid_witness: success.witness.valid,
            basis: success.witness.basis,
            steps: success
                .context
                .steps
                .into_iter()
                .map(|step| StepDto {
                    order: step.order,
                    name: step.name,
                    status: step.status,
                    detail: step.detail,
                })
                .collect(),
        },
    }
}

impl IntoResponse for AnalysisFailure {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (
            status,
            Json(ErrorResponse {
                ok: false,
                error: ErrorBody {
                    category: self.category,
                    message: self.message,
                    case_id: self.context.case_id,
                    run_id: self.context.run_id,
                    input_id: self.context.input_id,
                    details: self.details,
                    steps: self
                        .context
                        .steps
                        .into_iter()
                        .map(|step| StepDto {
                            order: step.order,
                            name: step.name,
                            status: step.status,
                            detail: step.detail,
                        })
                        .collect(),
                },
            }),
        )
            .into_response()
    }
}

fn simple_error(status: StatusCode, category: &str, message: &str) -> Response {
    (
        status,
        Json(json!({
            "ok": false,
            "error": {
                "category": category,
                "message": message
            }
        })),
    )
        .into_response()
}

async fn not_found() -> Response {
    simple_error(
        StatusCode::NOT_FOUND,
        "ROUTE_NOT_FOUND",
        "the requested route does not exist",
    )
}
