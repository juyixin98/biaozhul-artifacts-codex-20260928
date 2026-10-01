use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum VertexInput {
    Number(i64),
    Text(String),
}

impl VertexInput {
    pub fn into_label(self) -> String {
        match self {
            VertexInput::Number(value) => value.to_string(),
            VertexInput::Text(value) => value,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EdgeInput {
    Pair(VertexInput, VertexInput),
    Object { from: VertexInput, to: VertexInput },
}

impl EdgeInput {
    pub fn into_pair(self) -> (String, String) {
        match self {
            EdgeInput::Pair(left, right) => (left.into_label(), right.into_label()),
            EdgeInput::Object { from, to } => (from.into_label(), to.into_label()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphInput {
    #[serde(default)]
    pub vertices: Option<Vec<VertexInput>>,
    #[serde(default)]
    pub edges: Vec<EdgeInput>,
}

impl GraphInput {
    pub fn into_parts(self) -> (Option<Vec<String>>, Vec<(String, String)>) {
        let vertices = self
            .vertices
            .map(|values| values.into_iter().map(VertexInput::into_label).collect());
        let edges = self.edges.into_iter().map(EdgeInput::into_pair).collect();
        (vertices, edges)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnalyzeRequest {
    pub run_id: Option<String>,
    pub graph: GraphInput,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VerifyOrderRequest {
    pub run_id: Option<String>,
    pub graph: GraphInput,
    pub order: Vec<VertexInput>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphSummary {
    pub vertex_count: usize,
    pub edge_count: usize,
    pub duplicate_edge_count: usize,
    pub vertices: Vec<String>,
    pub normalized_edges: Vec<[String; 2]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckReport {
    pub valid: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrderCheckReport {
    pub valid: bool,
    pub checked_order: Vec<String>,
    pub first_failure: Option<OrderFailureDto>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrderFailureDto {
    pub position: usize,
    pub vertex: String,
    pub later_neighbors: Vec<String>,
    pub missing_edges: Vec<[String; 2]>,
    pub failure_category: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct WitnessDto {
    pub kind: &'static str,
    pub cycle_length: usize,
    pub cycle: Vec<String>,
    pub cycle_edges: Vec<[String; 2]>,
    pub induced: bool,
    pub independent_check: CheckReport,
}
