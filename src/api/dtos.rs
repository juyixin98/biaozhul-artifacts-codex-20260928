use serde::{Deserialize, Serialize};

use crate::algorithm::{EdgeDeletionReport, DecompositionReport};
use crate::diagnostics::RequestDiagnostics;
use crate::graph::{GraphInput, QueueOrder};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecomposeRequest {
    #[serde(flatten)]
    pub graph: GraphInput,
    #[serde(default)]
    pub include_events: Option<bool>,
    #[serde(default)]
    pub queue_order: QueueOrder,
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeleteEdgeRequest {
    #[serde(flatten)]
    pub graph: GraphInput,
    pub edge_id: usize,
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecomposeResponse {
    pub report: DecompositionReport,
    pub diagnostics: RequestDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeleteEdgeResponse {
    pub report: EdgeDeletionReport,
    pub diagnostics: RequestDiagnostics,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WitnessCheckRequest {
    pub report: DecompositionReport,
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WitnessCheckResponse {
    pub valid: bool,
    pub diagnostics: RequestDiagnostics,
}

