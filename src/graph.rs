//! The immutable simple-graph model and request normalization.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

pub type Node = usize;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MultipleEdgePolicy {
    /// Collapse repeated undirected declarations into one simple edge and retain multiplicity.
    Deduplicate,
    /// Treat repeated declarations for the same unordered endpoint pair as invalid input.
    Reject,
}

impl Default for MultipleEdgePolicy {
    fn default() -> Self {
        Self::Deduplicate
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NodeInput {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EdgeInput {
    #[serde(default)]
    pub id: Option<String>,
    pub u: String,
    pub v: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct GraphInput {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub multiple_edge_policy: MultipleEdgePolicy,
    #[serde(default)]
    pub nodes: Vec<NodeInput>,
    #[serde(default)]
    pub edges: Vec<EdgeInput>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Edge {
    pub id: String,
    pub u: Node,
    pub v: Node,
    pub multiplicity: usize,
}

#[derive(Debug, Clone)]
pub struct Neighbor {
    pub node: Node,
    pub edge: usize,
}

#[derive(Debug, Clone)]
pub struct SimpleGraph {
    pub node_ids: Vec<String>,
    pub node_labels: Vec<Option<String>>,
    pub edges: Vec<Edge>,
    adjacency: Vec<Vec<Neighbor>>,
}

#[derive(Debug, Clone, Serialize, Default, PartialEq, Eq)]
pub struct NormalizationReport {
    pub declared_node_count: usize,
    pub declared_edge_count: usize,
    pub simple_node_count: usize,
    pub simple_edge_count: usize,
    pub removed_duplicate_edge_declarations: usize,
    pub collapsed_pairs: Vec<CollapsedPair>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CollapsedPair {
    pub edge_id: String,
    pub multiplicity: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub state: BTreeMap<String, String>,
}

impl Diagnostic {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            state: BTreeMap::new(),
        }
    }

    pub fn with(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.state.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct GraphError {
    pub status: u16,
    pub code: String,
    pub message: String,
    pub diagnostics: Vec<Diagnostic>,
}

impl GraphError {
    fn invalid(diagnostics: Vec<Diagnostic>) -> Self {
        Self {
            status: 422,
            code: "INVALID_GRAPH".to_string(),
            message: "graph input failed validation".to_string(),
            diagnostics,
        }
    }
}

#[derive(Debug, Clone)]
struct EdgeOccurrence {
    id: String,
    u: String,
    v: String,
}

impl SimpleGraph {
    pub fn from_input(input: GraphInput) -> Result<(SimpleGraph, NormalizationReport), GraphError> {
        let mut diagnostics = Vec::new();
        let mut node_by_id = HashMap::new();
        let mut node_ids = Vec::new();
        let mut node_labels = Vec::new();

        for (index, node) in input.nodes.iter().enumerate() {
            if node.id.is_empty() {
                diagnostics.push(
                    Diagnostic::new("EMPTY_NODE_ID", "node id must be non-empty")
                        .with("node_index", index.to_string()),
                );
                continue;
            }
            if node_by_id.insert(node.id.clone(), node_ids.len()).is_some() {
                diagnostics.push(
                    Diagnostic::new("DUPLICATE_NODE_ID", "node id is declared more than once")
                        .with("node_index", index.to_string())
                        .with("redacted_node_id", redact(&node.id)),
                );
                continue;
            }
            node_ids.push(node.id.clone());
            node_labels.push(node.label.clone());
        }

        let mut occurrences = Vec::with_capacity(input.edges.len());
        let mut ids_seen = HashSet::new();

        for (index, edge) in input.edges.iter().enumerate() {
            let generated_id = edge.id.clone().unwrap_or_else(|| format!("e{index}"));
            if !ids_seen.insert(generated_id.clone()) {
                diagnostics.push(
                    Diagnostic::new("DUPLICATE_EDGE_ID", "edge id is not unique")
                        .with("edge_index", index.to_string())
                        .with("redacted_edge_id", redact(&generated_id)),
                );
            }

            if edge.u == edge.v {
                diagnostics.push(
                    Diagnostic::new("SELF_LOOP", "self loops are not allowed in a simple graph")
                        .with("edge_index", index.to_string())
                        .with("edge_id", generated_id.clone())
                        .with("redacted_endpoint", redact(&edge.u)),
                );
            }

            if !node_by_id.contains_key(&edge.u) {
                diagnostics.push(
                    Diagnostic::new("UNKNOWN_ENDPOINT", "edge endpoint is not in nodes")
                        .with("edge_index", index.to_string())
                        .with("endpoint", "u")
                        .with("redacted_endpoint", redact(&edge.u)),
                );
            }
            if !node_by_id.contains_key(&edge.v) {
                diagnostics.push(
                    Diagnostic::new("UNKNOWN_ENDPOINT", "edge endpoint is not in nodes")
                        .with("edge_index", index.to_string())
                        .with("endpoint", "v")
                        .with("redacted_endpoint", redact(&edge.v)),
                );
            }

            occurrences.push(EdgeOccurrence {
                id: generated_id,
                u: edge.u.clone(),
                v: edge.v.clone(),
            });
        }

        let mut pairs: BTreeMap<(String, String), Vec<&EdgeOccurrence>> = BTreeMap::new();
        for occurrence in &occurrences {
            let (a, b) = if occurrence.u <= occurrence.v {
                (occurrence.u.clone(), occurrence.v.clone())
            } else {
                (occurrence.v.clone(), occurrence.u.clone())
            };
            pairs.entry((a, b)).or_default().push(occurrence);
        }

        if input.multiple_edge_policy == MultipleEdgePolicy::Reject {
            for (index, group) in pairs.values().enumerate().filter(|(_, g)| g.len() > 1) {
                let first = group[0];
                diagnostics.push(
                    Diagnostic::new(
                        "MULTIPLE_EDGE",
                        "multiple edges are forbidden by the declared reject policy",
                    )
                    .with("pair_index", index.to_string())
                    .with("edge_id", first.id.clone())
                    .with("declared_count", group.len().to_string())
                    .with("redacted_u", redact(&first.u))
                    .with("redacted_v", redact(&first.v)),
                );
            }
        }

        if !diagnostics.is_empty() {
            return Err(GraphError::invalid(diagnostics));
        }

        let mut edges = Vec::new();
        let mut adjacency = vec![Vec::new(); node_ids.len()];
        let mut collapsed_pairs = Vec::new();
        let mut removed_duplicates = 0;

        for ((u_id, v_id), group) in pairs {
            let first = &group[0];
            let u = node_by_id[&u_id];
            let v = node_by_id[&v_id];
            let edge_index = edges.len();
            let multiplicity = group.len();

            edges.push(Edge {
                id: first.id.clone(),
                u,
                v,
                multiplicity,
            });
            adjacency[u].push(Neighbor {
                node: v,
                edge: edge_index,
            });
            adjacency[v].push(Neighbor {
                node: u,
                edge: edge_index,
            });

            if multiplicity > 1 {
                removed_duplicates += multiplicity - 1;
                collapsed_pairs.push(CollapsedPair {
                    edge_id: first.id.clone(),
                    multiplicity,
                });
            }
        }

        for neighbors in &mut adjacency {
            neighbors.sort_by_key(|neighbor| neighbor.node);
        }

        let report = NormalizationReport {
            declared_node_count: input.nodes.len(),
            declared_edge_count: input.edges.len(),
            simple_node_count: node_ids.len(),
            simple_edge_count: edges.len(),
            removed_duplicate_edge_declarations: removed_duplicates,
            collapsed_pairs,
        };

        Ok((
            SimpleGraph {
                node_ids,
                node_labels,
                edges,
                adjacency,
            },
            report,
        ))
    }

    pub fn node_count(&self) -> usize {
        self.node_ids.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn neighbors(&self, node: Node) -> &[Neighbor] {
        &self.adjacency[node]
    }

    pub fn edge_endpoints(&self, edge: usize) -> (Node, Node) {
        (self.edges[edge].u, self.edges[edge].v)
    }

    pub fn common_neighbors(&self, u: Node, v: Node) -> Vec<Node> {
        let left: HashSet<Node> = self.adjacency[u].iter().map(|item| item.node).collect();
        let right: HashSet<Node> = self.adjacency[v].iter().map(|item| item.node).collect();
        let mut shared: Vec<Node> = left.intersection(&right).copied().collect();
        shared.sort_unstable();
        shared
    }

    pub fn edge_support(&self, edge: usize) -> usize {
        let (u, v) = self.edge_endpoints(edge);
        self.common_neighbors(u, v).len()
    }

    pub fn triangle_count(&self) -> usize {
        let mut total = 0;
        for edge in &self.edges {
            total += self.common_neighbors(edge.u, edge.v).len();
        }
        total / 3
    }
}

fn redact(value: &str) -> String {
    if value.is_empty() {
        return "<empty>".to_string();
    }
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    format!("id_{:016x}", hash)
}

pub fn redact_public(value: &str) -> String {
    redact(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str) -> NodeInput {
        NodeInput {
            id: id.to_string(),
            label: None,
        }
    }

    #[test]
    fn normalizes_multiple_edges_by_default() {
        let input = GraphInput {
            nodes: vec![node("a"), node("b")],
            edges: vec![
                EdgeInput {
                    id: None,
                    u: "a".into(),
                    v: "b".into(),
                },
                EdgeInput {
                    id: None,
                    u: "b".into(),
                    v: "a".into(),
                },
            ],
            ..Default::default()
        };

        let (graph, report) = SimpleGraph::from_input(input).unwrap();
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.edges[0].multiplicity, 2);
        assert_eq!(report.removed_duplicate_edge_declarations, 1);
    }

    #[test]
    fn rejects_self_loops_with_a_specific_code() {
        let input = GraphInput {
            nodes: vec![node("a")],
            edges: vec![EdgeInput {
                id: Some("loop".into()),
                u: "a".into(),
                v: "a".into(),
            }],
            ..Default::default()
        };

        let error = SimpleGraph::from_input(input).unwrap_err();
        assert_eq!(error.status, 422);
        assert_eq!(error.diagnostics[0].code, "SELF_LOOP");
    }
}
