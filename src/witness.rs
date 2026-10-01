//! Independent checks for witness subgraphs returned by the decomposition.

use crate::graph::{Diagnostic, Node, SimpleGraph};
use crate::kernel::Decomposition;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VerificationReport {
    pub accepted: bool,
    pub diagnostics: Vec<Diagnostic>,
}

pub fn verify_decomposition(
    graph: &SimpleGraph,
    decomposition: &Decomposition,
) -> VerificationReport {
    let mut diagnostics = Vec::new();
    let witnesses_by_id: HashMap<&str, &crate::kernel::Witness> = decomposition
        .witnesses
        .iter()
        .map(|witness| (witness.witness_id.as_str(), witness))
        .collect();
    let edges_by_id: HashMap<&str, usize> = graph
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| (edge.id.as_str(), index))
        .collect();

    if decomposition.edges.len() != graph.edge_count() {
        diagnostics.push(
            Diagnostic::new(
                "WITNESS_EDGE_COUNT_MISMATCH",
                "decomposition must return exactly one result per normalized edge",
            )
            .with("expected", graph.edge_count().to_string())
            .with("actual", decomposition.edges.len().to_string()),
        );
    }

    let mut owner: HashMap<usize, String> = HashMap::new();
    for edge_result in &decomposition.edges {
        let Some(&edge) = edges_by_id.get(edge_result.edge_id.as_str()) else {
            diagnostics.push(
                Diagnostic::new("UNKNOWN_RESULT_EDGE", "result edge is absent from graph")
                    .with("redacted_edge_id", edge_result.edge_id.clone()),
            );
            continue;
        };
        if edge_result.level != decomposition.levels[edge] {
            diagnostics.push(
                Diagnostic::new(
                    "EDGE_LEVEL_INCONSISTENT",
                    "edge list and level vector disagree",
                )
                .with("edge_id", edge_result.edge_id.clone())
                .with("list_level", edge_result.level.to_string())
                .with("kernel_level", decomposition.levels[edge].to_string()),
            );
        }
        if owner.insert(edge, edge_result.witness_id.clone()).is_some() {
            diagnostics.push(
                Diagnostic::new(
                    "DUPLICATE_EDGE_RESULT",
                    "edge appears more than once in output",
                )
                .with("edge_id", edge_result.edge_id.clone()),
            );
        }

        let Some(witness) = witnesses_by_id.get(edge_result.witness_id.as_str()) else {
            diagnostics.push(
                Diagnostic::new("MISSING_WITNESS", "edge points to an absent witness")
                    .with("edge_id", edge_result.edge_id.clone())
                    .with("witness_id", edge_result.witness_id.clone()),
            );
            continue;
        };
        if witness.level != edge_result.level {
            diagnostics.push(
                Diagnostic::new(
                    "WITNESS_LEVEL_MISMATCH",
                    "witness level does not equal edge level",
                )
                .with("edge_id", edge_result.edge_id.clone())
                .with("edge_level", edge_result.level.to_string())
                .with("witness_level", witness.level.to_string()),
            );
        }
        if !witness.edge_ids.contains(&edge_result.edge_id) {
            diagnostics.push(
                Diagnostic::new(
                    "EDGE_ABSENT_FROM_WITNESS",
                    "edge is not contained in its witness",
                )
                .with("edge_id", edge_result.edge_id.clone())
                .with("witness_id", witness.witness_id.clone()),
            );
        }
    }

    for witness in &decomposition.witnesses {
        let mut edge_indices = Vec::new();
        let mut node_indices = HashSet::new();
        for edge_id in &witness.edge_ids {
            let Some(&edge) = edges_by_id.get(edge_id.as_str()) else {
                diagnostics.push(
                    Diagnostic::new("WITNESS_UNKNOWN_EDGE", "witness contains an unknown edge")
                        .with("witness_id", witness.witness_id.clone())
                        .with("redacted_edge_id", edge_id.clone()),
                );
                continue;
            };
            edge_indices.push(edge);
            node_indices.insert(graph.edges[edge].u);
            node_indices.insert(graph.edges[edge].v);

            if decomposition.levels[edge] < witness.level {
                diagnostics.push(
                    Diagnostic::new(
                        "WITNESS_LOW_LEVEL_EDGE",
                        "witness contains an edge below its claimed truss level",
                    )
                    .with("witness_id", witness.witness_id.clone())
                    .with("edge_id", edge_id.clone())
                    .with("edge_level", decomposition.levels[edge].to_string())
                    .with("witness_level", witness.level.to_string()),
                );
            }
        }

        let edge_set: HashSet<usize> = edge_indices.iter().copied().collect();
        if edge_set.len() != witness.edge_ids.len() {
            diagnostics.push(
                Diagnostic::new("WITNESS_DUPLICATE_EDGE", "witness contains duplicate edges")
                    .with("witness_id", witness.witness_id.clone()),
            );
        }

        let expected_nodes: HashSet<String> = node_indices
            .iter()
            .map(|node| graph.node_ids[*node].clone())
            .collect();
        let actual_nodes: HashSet<String> = witness.node_ids.iter().cloned().collect();
        if expected_nodes != actual_nodes {
            diagnostics.push(
                Diagnostic::new(
                    "WITNESS_NODE_SET_MISMATCH",
                    "witness nodes must be exactly the endpoints of its edges",
                )
                .with("witness_id", witness.witness_id.clone()),
            );
        }

        if !is_connected(graph, &edge_set) {
            diagnostics.push(
                Diagnostic::new("WITNESS_DISCONNECTED", "witness subgraph is not connected")
                    .with("witness_id", witness.witness_id.clone()),
            );
        }

        let min_support = edge_indices
            .iter()
            .map(|&edge| witness_edge_support(graph, edge, &edge_set))
            .min();
        let required_support = witness.level.saturating_sub(2);
        if min_support != Some(witness.minimum_support) {
            diagnostics.push(
                Diagnostic::new(
                    "WITNESS_SUPPORT_REPORT_MISMATCH",
                    "reported minimum support does not match the witness graph",
                )
                .with("witness_id", witness.witness_id.clone())
                .with("reported", witness.minimum_support.to_string())
                .with(
                    "actual",
                    min_support
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "none".into()),
                ),
            );
        }
        if min_support.is_some_and(|support| support < required_support) {
            diagnostics.push(
                Diagnostic::new(
                    "WITNESS_SUPPORT_TOO_LOW",
                    "witness fails the minimum edge support threshold",
                )
                .with("witness_id", witness.witness_id.clone())
                .with("minimum_support", min_support.unwrap().to_string())
                .with("required_support", required_support.to_string()),
            );
        }

        let exact_edges: Vec<usize> = edge_indices
            .iter()
            .copied()
            .filter(|edge| decomposition.levels[*edge] == witness.level)
            .collect();
        if exact_edges.is_empty() {
            diagnostics.push(
                Diagnostic::new(
                    "WITNESS_WITHOUT_LEVEL_EDGE",
                    "witness must contain at least one edge at its exact level",
                )
                .with("witness_id", witness.witness_id.clone()),
            );
        } else {
            let layer_edges: HashSet<usize> = (0..graph.edge_count())
                .filter(|edge| decomposition.levels[*edge] >= witness.level)
                .collect();
            let next_support_limit = witness.level.saturating_sub(1);
            let all_survive_next = exact_edges
                .iter()
                .all(|edge| witness_edge_support(graph, *edge, &layer_edges) >= next_support_limit);
            if all_survive_next {
                diagnostics.push(
                    Diagnostic::new(
                        "WITNESS_LEVEL_NOT_MAXIMAL",
                        "every level edge appears to belong to a higher truss",
                    )
                    .with("witness_id", witness.witness_id.clone())
                    .with("witness_level", witness.level.to_string()),
                );
            }
        }
    }

    VerificationReport {
        accepted: diagnostics.is_empty(),
        diagnostics,
    }
}

fn is_connected(graph: &SimpleGraph, edge_set: &HashSet<usize>) -> bool {
    if edge_set.is_empty() {
        return false;
    }
    let start = graph.edges[*edge_set.iter().next().unwrap()].u;
    let mut seen = HashSet::from([start]);
    let mut stack = vec![start];

    while let Some(node) = stack.pop() {
        for neighbor in graph.neighbors(node) {
            if edge_set.contains(&neighbor.edge) && seen.insert(neighbor.node) {
                stack.push(neighbor.node);
            }
        }
    }

    let mut endpoints = HashSet::new();
    for edge in edge_set {
        endpoints.insert(graph.edges[*edge].u);
        endpoints.insert(graph.edges[*edge].v);
    }
    seen == endpoints
}

fn witness_edge_support(graph: &SimpleGraph, edge: usize, edge_set: &HashSet<usize>) -> usize {
    let (u, v) = graph.edge_endpoints(edge);
    common_neighbors_in_edgeset(graph, u, v, edge_set).len()
}

fn common_neighbors_in_edgeset(
    graph: &SimpleGraph,
    u: Node,
    v: Node,
    edge_set: &HashSet<usize>,
) -> Vec<Node> {
    let u_neighbors: HashSet<Node> = graph
        .neighbors(u)
        .iter()
        .filter(|item| edge_set.contains(&item.edge))
        .map(|item| item.node)
        .collect();
    let v_neighbors: HashSet<Node> = graph
        .neighbors(v)
        .iter()
        .filter(|item| edge_set.contains(&item.edge))
        .map(|item| item.node)
        .collect();
    u_neighbors.intersection(&v_neighbors).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeInput, GraphInput, NodeInput, SimpleGraph};
    use crate::kernel::{PeelingOrder, decompose};

    #[test]
    fn accepts_triangle_witness() {
        let input = GraphInput {
            nodes: vec![
                NodeInput {
                    id: "a".into(),
                    label: None,
                },
                NodeInput {
                    id: "b".into(),
                    label: None,
                },
                NodeInput {
                    id: "c".into(),
                    label: None,
                },
            ],
            edges: vec![
                EdgeInput {
                    id: Some("ab".into()),
                    u: "a".into(),
                    v: "b".into(),
                },
                EdgeInput {
                    id: Some("bc".into()),
                    u: "b".into(),
                    v: "c".into(),
                },
                EdgeInput {
                    id: Some("ca".into()),
                    u: "c".into(),
                    v: "a".into(),
                },
            ],
            ..Default::default()
        };
        let (graph, _) = SimpleGraph::from_input(input).unwrap();
        let decomposition = decompose(&graph, PeelingOrder::Declaration).unwrap();
        let report = verify_decomposition(&graph, &decomposition);
        assert!(report.accepted, "{report:?}");
    }
}
