use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

use crate::graph::BuildError;
use crate::kernel::AnalysisError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    InputError,
    StateConflict,
    ResourceExhausted,
    ComputationFailure,
    NotFound,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub error: ErrorEnvelope,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorEnvelope {
    pub kind: ErrorKind,
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
}

#[derive(Debug)]
pub enum ApiError {
    Input(&'static str, String),
    StateConflict(&'static str, String),
    ResourceExhausted(&'static str, String),
    ComputationFailure(&'static str, String),
    NotFound(&'static str, String),
}

impl ApiError {
    pub fn kind(&self) -> ErrorKind {
        match self {
            ApiError::Input(..) => ErrorKind::InputError,
            ApiError::StateConflict(..) => ErrorKind::StateConflict,
            ApiError::ResourceExhausted(..) => ErrorKind::ResourceExhausted,
            ApiError::ComputationFailure(..) => ErrorKind::ComputationFailure,
            ApiError::NotFound(..) => ErrorKind::NotFound,
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            ApiError::Input(code, _)
            | ApiError::StateConflict(code, _)
            | ApiError::ResourceExhausted(code, _)
            | ApiError::ComputationFailure(code, _)
            | ApiError::NotFound(code, _) => code,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            ApiError::Input(_, message)
            | ApiError::StateConflict(_, message)
            | ApiError::ResourceExhausted(_, message)
            | ApiError::ComputationFailure(_, message)
            | ApiError::NotFound(_, message) => message,
        }
    }

    pub fn with_run_id(self, run_id: Option<String>) -> (Option<String>, Self) {
        (run_id, self)
    }

    pub fn from_build(error: BuildError) -> Self {
        match error {
            BuildError::InputError(message) => ApiError::Input("invalid_graph", message),
            BuildError::StateConflict(message) => {
                ApiError::StateConflict("duplicate_vertex_id", message)
            }
            BuildError::ResourceExhausted(message) => {
                ApiError::ResourceExhausted("graph_too_large", message)
            }
        }
    }

    pub fn from_analysis(error: AnalysisError) -> Self {
        match error {
            AnalysisError::ResourceExhausted(message) => {
                ApiError::ResourceExhausted("analysis_budget_exhausted", message)
            }
            AnalysisError::ComputationFailure(message) => {
                ApiError::ComputationFailure("analysis_invariant_failed", message)
            }
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.kind() {
            ErrorKind::InputError => StatusCode::BAD_REQUEST,
            ErrorKind::StateConflict => StatusCode::CONFLICT,
            ErrorKind::ResourceExhausted => StatusCode::PAYLOAD_TOO_LARGE,
            ErrorKind::ComputationFailure => StatusCode::UNPROCESSABLE_ENTITY,
            ErrorKind::NotFound => StatusCode::NOT_FOUND,
        };
        let body = ErrorBody {
            error: ErrorEnvelope {
                kind: self.kind(),
                code: self.code(),
                message: self.message().to_owned(),
                run_id: None,
            },
        };
        (status, Json(body)).into_response()
    }
}

pub type ApiResult<T> = Result<T, ApiError>;
