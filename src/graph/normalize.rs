use std::collections::HashSet;

use super::{Endpoints, GraphInput, InputEdge, MultiEdgePolicy, SimpleGraph, Vertex};
use crate::diagnostics::{ErrorCategory, ServiceError};

#[derive(Debug, Clone)]
pub struct NormalizedGraph {
    pub graph: SimpleGraph,
    pub declared_vertex_count: usize,
    pub duplicate_edge_count: usize,
    pub rejected_self_loops: Vec<InputEdge>,
    pub input_edge_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DuplicateRecord {
    index: usize,
    u: Vertex,
    v: Vertex,
}

pub fn normalize_input(input: &GraphInput) -> Result<NormalizedGraph, ServiceError> {
    let declared_vertex_count = match &input.vertices {
        Some(vertices) => {
            let distinct: HashSet<Vertex> = vertices.iter().copied().collect();
            if distinct.len() != vertices.len() {
                return Err(ServiceError::invalid_graph(
                    ErrorCategory::InvalidGraph,
                    "vertices must be unique",
                    serde_json::json!({
                        "declared_vertex_count": vertices.len(),
                        "distinct_vertex_count": distinct.len(),
                    }),
                ));
            }
            vertices.iter().copied().max().map_or(0, |max| max + 1)
        }
        None => input
            .edges
            .iter()
            .flat_map(|edge| [edge.u, edge.v])
            .max()
            .map_or(0, |max| max + 1),
    };

    let mut seen_undirected: HashSet<(Vertex, Vertex)> = HashSet::new();
    let mut duplicates = Vec::new();
    let mut self_loops = Vec::new();
    let mut endpoints = Vec::new();

    for (index, edge) in input.edges.iter().enumerate() {
        if edge.u == edge.v {
            self_loops.push(*edge);
            continue;
        }
        let canonical = Endpoints::new(edge.u, edge.v);
        let key = (canonical.u, canonical.v);
        if seen_undirected.contains(&key) {
            duplicates.push(DuplicateRecord {
                index,
                u: edge.u,
                v: edge.v,
            });
            continue;
        }
        seen_undirected.insert(key);
        if canonical.u >= declared_vertex_count || canonical.v >= declared_vertex_count {
            return Err(ServiceError::invalid_graph(
                ErrorCategory::UnknownVertex,
                "edge endpoint is outside the declared vertex set",
                serde_json::json!({
                    "edge_index": index,
                    "declared_vertex_count": declared_vertex_count,
                    "u": edge.u,
                    "v": edge.v,
                }),
            ));
        }
        endpoints.push(canonical);
    }

    if !self_loops.is_empty() {
        return Err(ServiceError::invalid_graph(
            ErrorCategory::SelfLoop,
            "self loops are not allowed in an undirected simple graph",
            serde_json::json!({
                "self_loop_count": self_loops.len(),
                "edge_indexes": self_loops.iter().map(|edge| input.edges.iter().position(|candidate| candidate == edge).unwrap_or(usize::MAX)).collect::<Vec<_>>(),
            }),
        ));
    }

    if matches!(input.multi_edge_policy, MultiEdgePolicy::Reject) && !duplicates.is_empty() {
        return Err(ServiceError::invalid_graph(
            ErrorCategory::DuplicateEdge,
            "parallel/multi-edges were found and policy is reject",
            serde_json::json!({
                "duplicate_edge_count": duplicates.len(),
                "edge_indexes": duplicates.iter().map(|edge| edge.index).collect::<Vec<_>>(),
            }),
        ));
    }

    let graph = SimpleGraph::from_endpoints(declared_vertex_count, endpoints);

    Ok(NormalizedGraph {
        graph,
        declared_vertex_count,
        duplicate_edge_count: duplicates.len(),
        rejected_self_loops: self_loops,
        input_edge_count: input.edges.len(),
    })
}
