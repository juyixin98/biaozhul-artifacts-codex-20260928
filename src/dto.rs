use serde::{Deserialize, Serialize};

use crate::model::LabeledGraph;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GraphRequest {
    pub graph: GraphDto,
    pub case_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GraphDto {
    pub vertices: Vec<String>,
    #[serde(default)]
    pub edges: Vec<EdgeDto>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EdgeDto {
    pub from: String,
    pub to: String,
}

impl From<GraphDto> for LabeledGraph {
    fn from(value: GraphDto) -> Self {
        Self {
            vertices: value.vertices,
            edges: value
                .edges
                .into_iter()
                .map(|edge| (edge.from, edge.to))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ErrorResponse {
    pub ok: bool,
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ErrorBody {
    pub category: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub case_id: Option<String>,
    pub run_id: String,
    pub input_id: String,
    pub details: serde_json::Value,
    pub steps: Vec<StepDto>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SuccessResponse {
    pub ok: bool,
    pub result: PathCoverResult,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PathCoverResult {
    pub case_id: Option<String>,
    pub run_id: String,
    pub input_id: String,
    pub version: String,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub minimum_path_count: usize,
    pub matching_size: usize,
    pub paths: Vec<Vec<String>>,
    pub matching_pairs: Vec<LabeledEdgeDto>,
    pub sources: Vec<String>,
    pub sinks: Vec<String>,
    pub diagnostics: DiagnosticsDto,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LabeledEdgeDto {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticsDto {
    pub valid_witness: bool,
    pub basis: Vec<String>,
    pub steps: Vec<StepDto>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct StepDto {
    pub order: usize,
    pub name: String,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FixtureSummary {
    pub name: String,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub expected_category: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HealthResponse {
    pub ok: bool,
    pub service: String,
    pub version: String,
    pub bind_address: String,
    pub max_vertices: usize,
    pub max_edges: usize,
}
