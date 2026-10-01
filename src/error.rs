use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::ser::{SerializeStruct, Serializer};
use serde::Serialize;

pub const MAX_NODE_COUNT: usize = 1024;
pub const MAX_EDGE_COUNT: usize = 4096;
pub const MAX_ID_LEN: usize = 128;
pub const MAX_ACTIVE_JOBS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputErrorCode {
    InvalidJson,
    EmptyEdgeId,
    DuplicateEdgeId,
    EdgeIdTooLong,
    NodeCountOutOfRange,
    EdgeCountOutOfRange,
    EndpointOutOfRange,
    SourceEqualsSink,
    CapacityOutOfRange,
    CertificateNodeOutOfRange,
    CertificateCutSideOverlap,
}

impl InputErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::EmptyEdgeId => "empty_edge_id",
            Self::DuplicateEdgeId => "duplicate_edge_id",
            Self::EdgeIdTooLong => "edge_id_too_long",
            Self::NodeCountOutOfRange => "node_count_out_of_range",
            Self::EdgeCountOutOfRange => "edge_count_out_of_range",
            Self::EndpointOutOfRange => "endpoint_out_out_of_range",
            Self::SourceEqualsSink => "source_equals_sink",
            Self::CapacityOutOfRange => "capacity_out_of_range",
            Self::CertificateNodeOutOfRange => "certificate_node_out_of_range",
            Self::CertificateCutSideOverlap => "certificate_cut_side_overlap",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceErrorCode {
    GraphTooLarge,
    JobRegistryFull,
}

impl ResourceErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GraphTooLarge => "graph_too_large",
            Self::JobRegistryFull => "job_registry_full",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputationErrorCode {
    CapacityOverflow,
    WitnessCheckFailed,
    InconsistentSolverState,
    InternalStatePoisoned,
}

impl ComputationErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CapacityOverflow => "capacity_overflow",
            Self::WitnessCheckFailed => "witness_check_failed",
            Self::InconsistentSolverState => "inconsistent_solver_state",
            Self::InternalStatePoisoned => "internal_state_poisoned",
        }
    }
}

#[derive(Debug, Clone)]
pub enum AppError {
    Input(InputErrorCode, String),
    NotFound(String),
    StateConflict(String),
    ResourceExhausted(ResourceErrorCode, String),
    ComputationFailed(ComputationErrorCode, String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(code, message) => write!(f, "{}: {message}", code.as_str()),
            Self::NotFound(message) => write!(f, "not_found: {message}"),
            Self::StateConflict(message) => write!(f, "state_conflict: {message}"),
            Self::ResourceExhausted(code, message) => {
                write!(f, "{}: {message}", code.as_str())
            }
            Self::ComputationFailed(code, message) => {
                write!(f, "{}: {message}", code.as_str())
            }
        }
    }
}

impl AppError {
    pub fn input(code: InputErrorCode, message: impl Into<String>) -> Self {
        Self::Input(code, message.into())
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::NotFound(message.into())
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::StateConflict(message.into())
    }

    pub fn resource(code: ResourceErrorCode, message: impl Into<String>) -> Self {
        Self::ResourceExhausted(code, message.into())
    }

    pub fn computation(code: ComputationErrorCode, message: impl Into<String>) -> Self {
        Self::ComputationFailed(code, message.into())
    }

    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Input(InputErrorCode::InvalidJson, _) => StatusCode::BAD_REQUEST,
            Self::Input(_, _) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::StateConflict(_) => StatusCode::CONFLICT,
            Self::ResourceExhausted(_, _) => StatusCode::SERVICE_UNAVAILABLE,
            Self::ComputationFailed(_, _) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn category(&self) -> &'static str {
        match self {
            Self::Input(_, _) => "input_error",
            Self::NotFound(_) => "not_found",
            Self::StateConflict(_) => "state_conflict",
            Self::ResourceExhausted(_, _) => "resource_exhausted",
            Self::ComputationFailed(_, _) => "computation_failed",
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(code, _) => code.as_str(),
            Self::NotFound(_) => "not_found",
            Self::StateConflict(_) => "state_conflict",
            Self::ResourceExhausted(code, _) => code.as_str(),
            Self::ComputationFailed(code, _) => code.as_str(),
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Input(_, message)
            | Self::NotFound(message)
            | Self::StateConflict(message)
            | Self::ResourceExhausted(_, message)
            | Self::ComputationFailed(_, message) => message,
        }
    }
}

impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("AppError", 3)?;
        state.serialize_field("category", &self.category())?;
        state.serialize_field("code", &self.code())?;
        state.serialize_field("message", self.message())?;
        state.end()
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status_code(), Json(self)).into_response()
    }
}

impl std::error::Error for AppError {}
