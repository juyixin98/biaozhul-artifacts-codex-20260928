use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    InvalidJson,
    InvalidGraph,
    SelfLoop,
    DuplicateEdge,
    UnknownVertex,
    EdgeNotFound,
    InvalidWitness,
}

impl ErrorCategory {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::InvalidGraph => "invalid_graph",
            Self::SelfLoop => "self_loop",
            Self::DuplicateEdge => "duplicate_edge",
            Self::UnknownVertex => "unknown_vertex",
            Self::EdgeNotFound => "edge_not_found",
            Self::InvalidWitness => "invalid_witness",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticEvent {
    pub stage: String,
    pub decision: String,
    pub state: serde_json::Value,
}

impl DiagnosticEvent {
    pub fn new(stage: impl Into<String>, decision: impl Into<String>, state: serde_json::Value) -> Self {
        Self {
            stage: stage.into(),
            decision: decision.into(),
            state,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestDiagnostics {
    pub request_id: String,
    pub accepted: bool,
    pub decision: String,
    pub state: serde_json::Value,
    pub events: Vec<DiagnosticEvent>,
}

impl RequestDiagnostics {
    #[must_use]
    pub fn accepted(decision: impl Into<String>, state: serde_json::Value) -> Self {
        Self::with_id(Uuid::new_v4().to_string(), true, decision, state)
    }

    #[must_use]
    pub fn with_id(
        request_id: impl Into<String>,
        accepted: bool,
        decision: impl Into<String>,
        state: serde_json::Value,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            accepted,
            decision: decision.into(),
            state,
            events: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
    pub diagnostics: RequestDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub category: ErrorCategory,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ServiceError {
    pub category: ErrorCategory,
    pub message: String,
    pub diagnostics: RequestDiagnostics,
    pub status: StatusCode,
}

impl ServiceError {
    pub fn invalid_graph(
        category: ErrorCategory,
        message: impl Into<String>,
        state: serde_json::Value,
    ) -> Self {
        Self::new(
            category,
            message,
            state,
            StatusCode::UNPROCESSABLE_ENTITY,
        )
    }

    pub fn invalid_json(message: impl Into<String>, state: serde_json::Value) -> Self {
        Self::new(ErrorCategory::InvalidJson, message, state, StatusCode::BAD_REQUEST)
    }

    pub fn edge_not_found(message: impl Into<String>, state: serde_json::Value) -> Self {
        Self::new(ErrorCategory::EdgeNotFound, message, state, StatusCode::NOT_FOUND)
    }

    pub fn invalid_witness(message: impl Into<String>, state: serde_json::Value) -> Self {
        Self::new(
            ErrorCategory::InvalidWitness,
            message,
            state,
            StatusCode::UNPROCESSABLE_ENTITY,
        )
    }

    #[must_use]
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.diagnostics.request_id = request_id.into();
        self
    }

    fn new(
        category: ErrorCategory,
        message: impl Into<String>,
        state: serde_json::Value,
        status: StatusCode,
    ) -> Self {
        Self {
            category,
            message: message.into(),
            diagnostics: RequestDiagnostics::with_id(
                Uuid::new_v4().to_string(),
                false,
                "rejected",
                state,
            ),
            status,
        }
    }
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.category.as_str(), self.message)
    }
}

impl std::error::Error for ServiceError {}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        let status = self.status;
        let body = ErrorResponse {
            error: ErrorBody {
                category: self.category,
                message: self.message,
            },
            diagnostics: self.diagnostics,
        };
        (status, Json(body)).into_response()
    }
}
