use crate::config::AppConfig;
use crate::model::{InvalidEdgeReason, ModelError};
use crate::store::{GraphStore, Snapshot, StoreError};
use crate::witness::{self, WitnessClaimInput, WitnessReport};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri},
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Extension, Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

static REQUEST_ID_HEADER: OnceLock<HeaderName> = OnceLock::new();

fn request_id_header() -> HeaderName {
    REQUEST_ID_HEADER
        .get_or_init(|| HeaderName::from_static("x-request-id"))
        .clone()
}

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<GraphStore>,
    pub config: AppConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiResponse<T: Serialize> {
    pub request_id: String,
    pub data: T,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorBody {
    pub request_id: String,
    pub error: ApiErrorDetail,
    pub uncertainties: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorDetail {
    pub code: &'static str,
    pub message: String,
    pub location: Option<String>,
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    location: Option<String>,
    uncertainties: Vec<String>,
}

impl ApiError {
    fn new(
        status: StatusCode,
        code: &'static str,
        message: impl Into<String>,
        location: Option<String>,
    ) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            location,
            uncertainties: Vec::new(),
        }
    }
}

impl From<StoreError> for ApiError {
    fn from(value: StoreError) -> Self {
        match value {
            StoreError::GraphAlreadyExists(id) => Self::new(
                StatusCode::CONFLICT,
                "graph_already_exists",
                format!("graph `{id}` already exists"),
                Some("graph_id".into()),
            ),
            StoreError::GraphNotFound(id) => Self::new(
                StatusCode::NOT_FOUND,
                "graph_not_found",
                format!("graph `{id}` does not exist"),
                Some("graph_id".into()),
            ),
            StoreError::VersionNotFound { graph_id, version } => Self::new(
                StatusCode::NOT_FOUND,
                "version_not_found",
                format!("graph `{graph_id}` has no version {version}"),
                Some("version".into()),
            ),
            StoreError::ExpectedVersionMismatch {
                graph_id,
                actual,
                expected,
            } => Self::new(
                StatusCode::CONFLICT,
                "expected_version_mismatch",
                format!(
                    "graph `{graph_id}` is at version {actual}, request expected version {expected}"
                ),
                Some("expected_version".into()),
            ),
            StoreError::Model(error) => error.into(),
            StoreError::WitnessRejected(report) => {
                tracing::error!(
                    failures = ?report.failures,
                    uncertainties = ?report.uncertainties,
                    "rejected internally inconsistent version before publication"
                );
                Self::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "witness_rejected",
                    "dynamic kernel produced a snapshot rejected by the independent witness",
                    Some("batch-updates".into()),
                )
            }
        }
    }
}

impl From<ModelError> for ApiError {
    fn from(value: ModelError) -> Self {
        match value {
            ModelError::DuplicateEdgeId(ref id) => Self::new(
                StatusCode::CONFLICT,
                "duplicate_edge_id",
                value.to_string(),
                Some(format!("operations[].edge.id={id}")),
            ),
            ModelError::EdgeNotFound(ref id) => Self::new(
                StatusCode::NOT_FOUND,
                "edge_not_found",
                value.to_string(),
                Some(format!("operations[].edge_id={id}")),
            ),
            ModelError::InvalidEdge {
                ref edge_id,
                reason: InvalidEdgeReason::EmptyId,
            } => Self::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_edge_id",
                value.to_string(),
                Some(format!("operations[].edge.id={edge_id}")),
            ),
            ModelError::InvalidEdge {
                ref edge_id,
                reason: InvalidEdgeReason::IdTooLong,
            } => Self::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "edge_id_too_long",
                value.to_string(),
                Some(format!("operations[].edge.id={edge_id}")),
            ),
            ModelError::EdgeLimitExceeded { limit } => Self::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "edge_limit_exceeded",
                value.to_string(),
                Some(format!("config.max_edges={limit}")),
            ),
            ModelError::ArithmeticOverflow => Self::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "arithmetic_overflow",
                value.to_string(),
                Some("snapshot.total_weight".into()),
            ),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let request_id = RequestId::fallback();
        let mut response = Json(ApiErrorBody {
            request_id: request_id.0.clone(),
            error: ApiErrorDetail {
                code: self.code,
                message: self.message,
                location: self.location,
            },
            uncertainties: self.uncertainties,
        })
        .into_response();
        *response.status_mut() = self.status;
        response
            .headers_mut()
            .insert(request_id_header(), request_id.into_header_value());
        response
    }
}

#[derive(Debug, Clone)]
pub struct RequestId(pub String);

impl RequestId {
    fn fallback() -> Self {
        Self("unknown-request".to_string())
    }

    fn into_header_value(self) -> HeaderValue {
        HeaderValue::from_str(&self.0).unwrap_or(HeaderValue::from_static("invalid-request-id"))
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateGraphRequest {
    pub graph_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum OperationRequest {
    AddEdge { edge: EdgeRequest },
    DeleteEdge { edge_id: String },
    SetWeight { edge_id: String, weight: i64 },
}

#[derive(Debug, Deserialize)]
pub struct EdgeRequest {
    pub id: String,
    pub u: i64,
    pub v: i64,
    pub weight: i64,
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub expected_version: Option<u64>,
    pub operations: Vec<OperationRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchResponse {
    pub published_version: u64,
    pub previous_version: u64,
    pub snapshot: SnapshotDto,
    pub steps: Vec<crate::kernel::StepReport>,
    pub witness: WitnessReport,
}

#[derive(Debug, Serialize)]
pub struct VersionListResponse {
    pub graph_id: String,
    pub versions: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SnapshotDto {
    pub graph_id: String,
    pub version: u64,
    pub vertices: Vec<i64>,
    pub edges: Vec<EdgeDto>,
    pub forest_edge_ids: Vec<String>,
    pub total_weight: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeDto {
    pub id: String,
    pub u: i64,
    pub v: i64,
    pub weight: i64,
}

impl From<&Snapshot> for SnapshotDto {
    fn from(snapshot: &Snapshot) -> Self {
        Self {
            graph_id: snapshot.graph_id.clone(),
            version: snapshot.version,
            vertices: snapshot.vertices.iter().copied().collect(),
            edges: snapshot
                .edges
                .values()
                .map(|edge| EdgeDto {
                    id: edge.id.clone(),
                    u: edge.u,
                    v: edge.v,
                    weight: edge.weight,
                })
                .collect(),
            forest_edge_ids: snapshot.forest_edge_ids.iter().cloned().collect(),
            total_weight: snapshot.total_weight.to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct WitnessRequest {
    pub snapshot: Option<SnapshotClaimDto>,
    pub claim: Option<WitnessClaimInput>,
}

#[derive(Debug, Deserialize)]
pub struct SnapshotClaimDto {
    pub vertices: BTreeSet<i64>,
    pub edges: std::collections::BTreeMap<String, WitnessEdgeDto>,
    pub forest_edge_ids: Vec<String>,
    pub total_weight: String,
}

#[derive(Debug, Deserialize)]
pub struct WitnessEdgeDto {
    pub u: i64,
    pub v: i64,
    pub weight: i64,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/graphs", post(create_graph))
        .route("/graphs/:graph_id/versions", get(list_versions))
        .route("/graphs/:graph_id/versions/:version", get(get_version))
        .route("/graphs/:graph_id/batch-updates", post(batch_update))
        .route("/witness-checks", post(witness_check))
        .layer(from_fn_with_state(state.clone(), request_id_layer))
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn request_id_layer(
    headers: HeaderMap,
    method: Method,
    uri: Uri,
    request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Response {
    let request_id = headers
        .get(request_id_header())
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 128
                && value
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
        .map(|value| RequestId(value.to_string()))
        .unwrap_or_else(generate_request_id);

    tracing::info!(
        request_id = %request_id.0,
        method = %method,
        path = %uri.path(),
        "request started"
    );

    let request_id_value = request_id.0.clone();
    let mut request = request;
    request.extensions_mut().insert(request_id);

    let mut response = next.run(request).await;
    response.headers_mut().insert(
        request_id_header(),
        request_id_value
            .parse()
            .unwrap_or_else(|_| HeaderValue::from_static("invalid-request-id")),
    );
    if response.status().is_client_error() || response.status().is_server_error() {
        let response_body = std::mem::take(response.body_mut());
        if let Ok(bytes) = axum::body::to_bytes(response_body, 1024 * 1024).await {
            match serde_json::from_slice::<serde_json::Value>(&bytes) {
                Ok(mut value) => {
                    value["request_id"] = serde_json::Value::String(request_id_value.clone());
                    let new_body = serde_json::to_vec(&value).unwrap_or_else(|_| bytes.to_vec());
                    response
                        .headers_mut()
                        .remove(axum::http::header::CONTENT_LENGTH);
                    *response.body_mut() = axum::body::Body::from(new_body);
                }
                Err(_) => {
                    *response.body_mut() = axum::body::Body::from(bytes);
                }
            }
        }
    }
    tracing::info!(
        request_id = %request_id_value,
        status = response.status().as_u16(),
        location = ?response.status().canonical_reason(),
        "request completed"
    );
    response
}

fn generate_request_id() -> RequestId {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    RequestId(format!("req-{nanos}-{counter}"))
}

async fn create_graph(
    State(state): State<AppState>,
    Extension(request_id): Extension<RequestId>,
    Json(request): Json<CreateGraphRequest>,
) -> Result<Json<ApiResponse<SnapshotDto>>, ApiError> {
    validate_graph_id(&request.graph_id, &state.config)?;
    let snapshot = state.store.create_graph(&request.graph_id, &state.config)?;
    tracing::info!(
        request_id = %request_id.0,
        graph_id = %snapshot.graph_id,
        version = snapshot.version,
        "published complete graph version"
    );
    Ok(Json(ApiResponse {
        request_id: request_id.0,
        data: SnapshotDto::from(&snapshot),
    }))
}

async fn list_versions(
    State(state): State<AppState>,
    Extension(request_id): Extension<RequestId>,
    Path(graph_id): Path<String>,
) -> Result<Json<ApiResponse<VersionListResponse>>, ApiError> {
    validate_graph_id(&graph_id, &state.config)?;
    let versions = state.store.list_versions(&graph_id)?;
    Ok(Json(ApiResponse {
        request_id: request_id.0,
        data: VersionListResponse { graph_id, versions },
    }))
}

async fn get_version(
    State(state): State<AppState>,
    Extension(request_id): Extension<RequestId>,
    Path((graph_id, version)): Path<(String, u64)>,
) -> Result<Json<ApiResponse<SnapshotDto>>, ApiError> {
    validate_graph_id(&graph_id, &state.config)?;
    let snapshot = state.store.get_version(&graph_id, version)?;
    Ok(Json(ApiResponse {
        request_id: request_id.0,
        data: SnapshotDto::from(&snapshot),
    }))
}

async fn batch_update(
    State(state): State<AppState>,
    Extension(request_id): Extension<RequestId>,
    Path(graph_id): Path<String>,
    Json(request): Json<BatchRequest>,
) -> Result<Json<ApiResponse<BatchResponse>>, ApiError> {
    validate_graph_id(&graph_id, &state.config)?;
    if request.operations.is_empty() {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "empty_batch",
            "batch operations must contain at least one operation",
            Some("operations".into()),
        ));
    }
    if request.operations.len() > state.config.max_operations_per_batch {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "batch_too_large",
            format!(
                "batch has {} operations; limit is {}",
                request.operations.len(),
                state.config.max_operations_per_batch
            ),
            Some("operations".into()),
        ));
    }

    let operations = request
        .operations
        .into_iter()
        .map(operation_from_request)
        .collect::<Result<Vec<_>, _>>()?;

    let previous = state.store.latest_version(&graph_id)?;
    let previous_version = previous.version;
    let mut steps = Vec::new();

    let snapshot = state.store.mutate(
        &graph_id,
        request.expected_version,
        &state.config,
        |kernel| {
            for (index, operation) in operations.clone().into_iter().enumerate() {
                let step = kernel.apply(index, operation)?;
                tracing::info!(
                    request_id = %request_id.0,
                    graph_id = %graph_id,
                    step_index = index,
                    operation = step.operation,
                    edge_id = %step.edge_id,
                    action = %step.action,
                    forest_added = ?step.forest_edge_added,
                    forest_removed = ?step.forest_edge_removed,
                    "applied maintenance operation"
                );
                steps.push(step);
            }
            Ok(())
        },
    )?;

    let witness_report = witness::verify_snapshot(&snapshot);
    if state.config.verify_after_batch && witness_report.status != "accepted" {
        tracing::error!(
            request_id = %request_id.0,
            graph_id = %graph_id,
            version = snapshot.version,
            failures = ?witness_report.failures,
            uncertainties = ?witness_report.uncertainties,
            "rejected internally inconsistent published result"
        );
        return Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "witness_rejected",
            "dynamic kernel produced a snapshot rejected by the independent witness",
            Some(format!("graphs/{}/versions/{}", graph_id, snapshot.version)),
        ));
    }

    tracing::info!(
        request_id = %request_id.0,
        graph_id = %graph_id,
        previous_version,
        published_version = snapshot.version,
        total_weight = %snapshot.total_weight,
        "published complete graph version"
    );

    Ok(Json(ApiResponse {
        request_id: request_id.0,
        data: BatchResponse {
            published_version: snapshot.version,
            previous_version,
            snapshot: SnapshotDto::from(&snapshot),
            steps,
            witness: witness_report,
        },
    }))
}

async fn witness_check(
    Extension(request_id): Extension<RequestId>,
    Json(request): Json<WitnessRequest>,
) -> Json<ApiResponse<WitnessReport>> {
    let report = if let Some(snapshot) = request.snapshot {
        let total_weight = snapshot.total_weight.parse::<i128>().unwrap_or(i128::MAX);
        let claim = WitnessClaimInput {
            vertices: snapshot.vertices,
            edges: snapshot
                .edges
                .into_iter()
                .map(|(id, edge)| {
                    (
                        id,
                        witness::WitnessEdge {
                            u: edge.u,
                            v: edge.v,
                            weight: edge.weight,
                        },
                    )
                })
                .collect(),
            forest_edge_ids: snapshot.forest_edge_ids.iter().cloned().collect(),
        };
        witness::verify_claim(&claim, Some(total_weight))
    } else if let Some(claim) = request.claim {
        witness::verify_claim(&claim, None)
    } else {
        witness::verify_claim(
            &WitnessClaimInput {
                vertices: BTreeSet::new(),
                edges: std::collections::BTreeMap::new(),
                forest_edge_ids: Vec::new(),
            },
            Some(0),
        )
    };

    tracing::info!(
        request_id = %request_id.0,
        status = report.status,
        failure_count = report.failures.len(),
        uncertainty_count = report.uncertainties.len(),
        "independent witness check completed"
    );

    Json(ApiResponse {
        request_id: request_id.0,
        data: report,
    })
}

fn operation_from_request(
    value: OperationRequest,
) -> Result<crate::kernel::MaintenanceOp, ApiError> {
    Ok(match value {
        OperationRequest::AddEdge { edge } => crate::kernel::MaintenanceOp::AddEdge(
            crate::model::Edge::new(edge.id, edge.u, edge.v, edge.weight),
        ),
        OperationRequest::DeleteEdge { edge_id } => {
            crate::kernel::MaintenanceOp::DeleteEdge { id: edge_id }
        }
        OperationRequest::SetWeight { edge_id, weight } => {
            crate::kernel::MaintenanceOp::SetWeight {
                id: edge_id,
                weight,
            }
        }
    })
}

fn validate_graph_id(graph_id: &str, config: &AppConfig) -> Result<(), ApiError> {
    if graph_id.is_empty()
        || graph_id.len() > config.max_graph_id_length
        || graph_id.chars().any(char::is_whitespace)
    {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_graph_id",
            "graph id must be non-empty, no whitespace, and within configured length",
            Some("graph_id".into()),
        ));
    }
    Ok(())
}
