use serde::{Deserialize, Serialize};

use crate::graph::{EdgeId, GraphSnapshot, Triangle, Vertex};
use crate::witness::Witness;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecompositionReport {
    pub vertex_count: usize,
    pub input_edge_count: usize,
    pub normalized_edge_count: usize,
    pub duplicate_edge_count: usize,
    pub triangle_count: usize,
    pub initial_total_support: usize,
    pub levels: Vec<EdgeTrussResult>,
    pub witnesses: Vec<Witness>,
    pub events: Vec<DecompositionEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeTrussResult {
    pub edge_id: EdgeId,
    pub u: Vertex,
    pub v: Vertex,
    pub initial_support: usize,
    pub truss_level: usize,
    pub witness_id: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecompositionEvent {
    pub stage_level: usize,
    pub edge_id: EdgeId,
    pub u: Vertex,
    pub v: Vertex,
    pub support_before_removal: usize,
    pub affected_triangle_count: usize,
    pub witness_id: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompositionStep {
    pub stage_level: usize,
    pub edge_id: EdgeId,
    pub support_before_removal: usize,
    pub affected_triangles: Vec<Triangle>,
    pub witness_id: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessAssignment {
    pub witness_id: usize,
    pub subgraph: GraphSnapshot,
}

