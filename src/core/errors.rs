use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Request values that can be rejected without starting the algorithm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InputError {
    EmptyNodeSet,
    MalformedJson {
        message: String,
    },
    SourceOutOfRange {
        source: usize,
        node_count: usize,
    },
    SinkOutOfRange {
        sink: usize,
        node_count: usize,
    },
    SourceEqualsSink {
        node: usize,
    },
    EdgeEndpointOutOfRange {
        index: usize,
        endpoint: String,
        node: usize,
        node_count: usize,
    },
    LimitExceeded(ResourceLimit),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimit {
    pub name: String,
    pub limit: u64,
    pub actual: u64,
}

/// Kernel and certificate failures. These are distinct from malformed input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KernelError {
    Canceled,
    ResourceExhausted(ResourceLimit),
    ArithmeticOverflow,
    WitnessInvalid { reason: WitnessFailure },
    InternalError { message: String },
}

/// Machine-readable failure category for an independently checked certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WitnessFailure {
    EdgeFlowCountMismatch {
        expected: usize,
        actual: usize,
    },
    FlowExceedsCapacity {
        edge_index: usize,
        capacity: u64,
        flow: u64,
    },
    FlowBelowZero {
        edge_index: usize,
        flow: i128,
    },
    BalanceViolation {
        node: usize,
        balance: i128,
    },
    SourceNetFlowMismatch {
        expected: u64,
        actual: i128,
    },
    SinkNetFlowMismatch {
        expected: u64,
        actual: i128,
    },
    CutNodeOutOfRange {
        node: usize,
        node_count: usize,
    },
    CutDuplicateNode {
        node: usize,
    },
    CutMissingSource {
        source: usize,
    },
    CutContainsSink {
        sink: usize,
    },
    CutDoesNotCoverAllNodes,
    CutCapacityMismatch {
        computed: u64,
        claimed: u64,
    },
    FlowCutMismatch {
        flow_value: u64,
        cut_capacity: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    InputError,
    StateConflict,
    ResourceExhausted,
    ComputationFailed,
    NotFound,
}

impl ErrorCategory {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCategory::InputError => "input_error",
            ErrorCategory::StateConflict => "state_conflict",
            ErrorCategory::ResourceExhausted => "resource_exhausted",
            ErrorCategory::ComputationFailed => "computation_failed",
            ErrorCategory::NotFound => "not_found",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ServiceError {
    InvalidInput(InputError),
    InvalidState {
        run_id: String,
        message: String,
    },
    ResourceExhausted(ResourceLimit),
    ComputationFailed(KernelError),
    JobNotFound {
        run_id: String,
    },
    RequestTooLarge {
        limit_bytes: usize,
        actual_bytes: usize,
    },
}

impl ServiceError {
    #[must_use]
    pub fn category(&self) -> ErrorCategory {
        match self {
            ServiceError::InvalidInput(_) => ErrorCategory::InputError,
            ServiceError::InvalidState { .. } => ErrorCategory::StateConflict,
            ServiceError::ResourceExhausted(_) => ErrorCategory::ResourceExhausted,
            ServiceError::ComputationFailed(_) => ErrorCategory::ComputationFailed,
            ServiceError::JobNotFound { .. } => ErrorCategory::NotFound,
            ServiceError::RequestTooLarge { .. } => ErrorCategory::ResourceExhausted,
        }
    }

    #[must_use]
    pub fn status_code(&self) -> StatusCode {
        match self {
            ServiceError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            ServiceError::InvalidState { .. } => StatusCode::CONFLICT,
            ServiceError::ResourceExhausted(_) => StatusCode::INSUFFICIENT_STORAGE,
            ServiceError::ComputationFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ServiceError::JobNotFound { .. } => StatusCode::NOT_FOUND,
            ServiceError::RequestTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
        }
    }
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        let category = self.category();
        let status = self.status_code();
        let run_id = match &self {
            ServiceError::InvalidState { run_id, .. } | ServiceError::JobNotFound { run_id } => {
                Some(run_id.clone())
            }
            _ => None,
        };
        let body = Json(json!({
            "run_id": run_id,
            "category": category,
            "error": self,
        }));
        (status, body).into_response()
    }
}

impl From<InputError> for ServiceError {
    fn from(value: InputError) -> Self {
        ServiceError::InvalidInput(value)
    }
}

impl From<KernelError> for ServiceError {
    fn from(value: KernelError) -> Self {
        match value {
            KernelError::ResourceExhausted(limit) => ServiceError::ResourceExhausted(limit),
            other => ServiceError::ComputationFailed(other),
        }
    }
}
