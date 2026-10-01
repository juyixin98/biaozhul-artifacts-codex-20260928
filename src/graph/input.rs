use serde::{Deserialize, Serialize};

use super::Vertex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MultiEdgePolicy {
    #[default]
    Deduplicate,
    Reject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphInput {
    #[serde(default)]
    pub vertices: Option<Vec<Vertex>>,
    #[serde(default)]
    pub edges: Vec<InputEdge>,
    #[serde(default)]
    pub multi_edge_policy: MultiEdgePolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputEdge {
    pub u: Vertex,
    pub v: Vertex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QueueOrder {
    #[default]
    Initial,
    ReverseInitial,
    Lifo,
}

