use serde::{Deserialize, Serialize};

use crate::graph::{EdgeId, SimpleGraph, Triangle};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeDeletionReport {
    pub removed_edge: EdgeId,
    pub triangle_count_before: usize,
    pub triangle_count_after: usize,
    pub affected_triangles: Vec<Triangle>,
    /// Support of every remaining edge after one decrement per affected triangle.
    pub support_after: Vec<usize>,
}

/// Delete an active edge and update support exactly once for each triangle that
/// contained it.
///
/// The triangle list is deduplicated before mutation. Each triangle is a
/// witness to a single support decrement for its two other edges, even if a
/// graph scan can reach the same triangle from more than one adjacency path.
pub fn delete_edge_once_per_triangle(
    graph: &mut SimpleGraph,
    edge: EdgeId,
) -> Result<EdgeDeletionReport, crate::diagnostics::ServiceError> {
    if !graph.is_active(edge) {
        return Err(crate::diagnostics::ServiceError::edge_not_found(
            "edge is absent or already inactive",
            serde_json::json!({"edge_id": edge, "active_edges": graph.active_edge_count()}),
        ));
    }

    let endpoints = graph.edge(edge).expect("active edge has endpoints");
    let triangles_before = graph.triangles();
    let mut support_after = initial_support(graph);

    // Canonical triangles naturally compare by (a,b,c), so this catches any
    // duplicate found by future adjacency traversal changes as well as current
    // implementation mistakes.
    let mut affected: Vec<Triangle> = triangles_before
        .into_iter()
        .filter(|triangle| {
            triangle.vertices().contains(&endpoints.u)
                && triangle.vertices().contains(&endpoints.v)
        })
        .collect();
    affected.sort_unstable();
    affected.dedup();

    graph.remove_edge(edge);

    // In an undirected simple graph, triangle edges are unique. Starting from
    // the deduped triangle set makes repeated support decrements impossible.
    for triangle in &affected {
        for (u, v) in triangle.edges() {
            if (u == endpoints.u && v == endpoints.v) || (u == endpoints.v && v == endpoints.u) {
                continue;
            }
            if let Some(other_edge) = graph.edge_id(u, v) {
                support_after[other_edge] = support_after[other_edge].saturating_sub(1);
            }
        }
    }

    let triangles_after_count = graph.triangles().len();

    Ok(EdgeDeletionReport {
        removed_edge: edge,
        triangle_count_before: affected.len() + triangles_after_count,
        triangle_count_after: triangles_after_count,
        affected_triangles: affected,
        support_after,
    })
}

#[must_use]
pub fn initial_support(graph: &SimpleGraph) -> Vec<usize> {
    let triangles = graph.triangles();
    let mut support = vec![0; graph.edges().len()];
    for triangle in triangles {
        for (u, v) in triangle.edges() {
            if let Some(edge) = graph.edge_id(u, v) {
                support[edge] += 1;
            }
        }
    }
    support
}
