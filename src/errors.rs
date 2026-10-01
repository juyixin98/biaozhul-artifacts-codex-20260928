use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use serde_json::Value;

pub const LIMIT_VERTICES: usize = 128;
pub const LIMIT_RAW_EDGES: usize = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    InputMalformed,
    ResourceExhausted,
    StateConflict,
    ComputationFailed,
}

#[derive(Debug)]
pub enum InputError {
    EmptyVertexName,
    DuplicateVertexName(String),
    MissingVertex(String),
    SelfLoop(String),
    TooManyVertices(usize),
    TooManyEdges(usize),
    InvalidOrderLength { expected: usize, actual: usize },
    DuplicateOrderVertex(String),
    UnknownOrderVertex(String),
}

impl InputError {
    pub fn category(&self) -> ErrorCategory {
        match self {
            InputError::TooManyVertices(_) | InputError::TooManyEdges(_) => {
                ErrorCategory::ResourceExhausted
            }
            _ => ErrorCategory::InputMalformed,
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            InputError::EmptyVertexName => "empty_vertex_name",
            InputError::DuplicateVertexName(_) => "duplicate_vertex_name",
            InputError::MissingVertex(_) => "edge_unknown_vertex",
            InputError::SelfLoop(_) => "self_loop",
            InputError::TooManyVertices(_) => "too_many_vertices",
            InputError::TooManyEdges(_) => "too_many_edges",
            InputError::InvalidOrderLength { .. } => "invalid_order_length",
            InputError::DuplicateOrderVertex(_) => "duplicate_order_vertex",
            InputError::UnknownOrderVertex(_) => "unknown_order_vertex",
        }
    }

    pub fn status(&self) -> StatusCode {
        match self.category() {
            ErrorCategory::ResourceExhausted => StatusCode::PAYLOAD_TOO_LARGE,
            _ => StatusCode::BAD_REQUEST,
        }
    }

    pub fn details(&self) -> Value {
        match self {
            InputError::EmptyVertexName => Value::Null,
            InputError::DuplicateVertexName(v) => serde_json::json!({ "vertex": v }),
            InputError::MissingVertex(v) => serde_json::json!({ "vertex": v }),
            InputError::SelfLoop(v) => serde_json::json!({ "vertex": v }),
            InputError::TooManyVertices(n) => serde_json::json!({
                "actual": n,
                "limit": LIMIT_VERTICES
            }),
            InputError::TooManyEdges(n) => serde_json::json!({
                "actual": n,
                "limit": LIMIT_RAW_EDGES
            }),
            InputError::InvalidOrderLength { expected, actual } => serde_json::json!({
                "expected": expected,
                "actual": actual
            }),
            InputError::DuplicateOrderVertex(v) => serde_json::json!({ "vertex": v }),
            InputError::UnknownOrderVertex(v) => serde_json::json!({ "vertex": v }),
        }
    }
}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputError::EmptyVertexName => write!(f, "vertex names must be non-empty"),
            InputError::DuplicateVertexName(v) => write!(f, "duplicate vertex name: {v}"),
            InputError::MissingVertex(v) => write!(f, "edge references missing vertex: {v}"),
            InputError::SelfLoop(v) => write!(f, "self loops are not supported: {v}"),
            InputError::TooManyVertices(n) => {
                write!(f, "vertex count {n} exceeds limit {LIMIT_VERTICES}")
            }
            InputError::TooManyEdges(n) => {
                write!(f, "raw edge count {n} exceeds limit {LIMIT_RAW_EDGES}")
            }
            InputError::InvalidOrderLength { expected, actual } => {
                write!(
                    f,
                    "order length {actual} must equal vertex count {expected}"
                )
            }
            InputError::DuplicateOrderVertex(v) => {
                write!(f, "order contains duplicate vertex: {v}")
            }
            InputError::UnknownOrderVertex(v) => {
                write!(f, "order contains unknown vertex: {v}")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub category: ErrorCategory,
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Value::is_null")]
    pub details: Value,
    #[serde(skip)]
    pub status: StatusCode,
    #[serde(skip)]
    pub run_id: Option<String>,
}

impl ApiError {
    pub fn input(error: InputError) -> Self {
        Self {
            category: error.category(),
            code: error.code().to_string(),
            message: error.to_string(),
            details: error.details(),
            status: error.status(),
            run_id: None,
        }
    }

    pub fn computation(message: String, details: Value) -> Self {
        Self {
            category: ErrorCategory::ComputationFailed,
            code: "computation_invariant_failed".to_string(),
            message,
            details,
            status: StatusCode::INTERNAL_SERVER_ERROR,
            run_id: None,
        }
    }

    pub fn state_conflict(code: &str, message: String, details: Value) -> Self {
        Self {
            category: ErrorCategory::StateConflict,
            code: code.to_string(),
            message,
            details,
            status: StatusCode::CONFLICT,
            run_id: None,
        }
    }

    pub fn with_run(mut self, run_id: String) -> Self {
        self.run_id = Some(run_id);
        self
    }
}

#[derive(Serialize)]
struct ErrorEnvelope {
    run_id: Option<String>,
    category: ErrorCategory,
    code: String,
    message: String,
    #[serde(skip_serializing_if = "Value::is_null")]
    details: Value,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorEnvelope {
            run_id: self.run_id,
            category: self.category,
            code: self.code,
            message: self.message.clone(),
            details: self.details.clone(),
        };
        (
            self.status,
            Json(serde_json::to_value(body).unwrap_or_default()),
        )
            .into_response()
    }
}

pub fn json_parse_error(error: serde_json::Error) -> ApiError {
    ApiError {
        category: ErrorCategory::InputMalformed,
        code: "invalid_json".to_string(),
        message: format!("request body is not valid JSON: {error}"),
        details: serde_json::json!({ "line": error.line(), "column": error.column() }),
        status: StatusCode::BAD_REQUEST,
        run_id: None,
    }
}
