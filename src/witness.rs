//! Independent witness checks for maintained minimum spanning forests.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::Serialize;

use crate::forest::Forest;
use crate::model::{Edge, EdgeId, Graph, Vertex};

/// One concrete witness failure category.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "category", rename_all = "snake_case")]
pub enum WitnessFailure {
    /// A reported forest edge is absent from the complete graph.
    MissingEdge(String),
    /// A reported forest edge is a self-loop.
    SelfLoopSelected(String),
    /// A self-loop is present in the graph but should not affect the forest.
    SelfLoopNotExcluded(String),
    /// Selected edges contain an undirected cycle.
    CycleDetected(String),
    /// Selected edges connect vertices that belong in different full-graph components.
    SpansDisconnectedComponents {
        /// Edge that connects two unreachable vertices.
        edge_id: String,
        /// First endpoint.
        u: String,
        /// Second endpoint.
        v: String,
    },
    /// A non-trivial full-graph component is not fully spanned.
    NotSpanning {
        /// Representative vertex of the component.
        component_root: String,
        /// Number of vertices in the full-graph component.
        graph_vertices: usize,
        /// Number reached by selected forest edges.
        forest_vertices: usize,
    },
    /// Maintained total differs from exact selected-edge sum.
    TotalMismatch {
        /// Total reported by the core.
        reported: i128,
        /// Total independently summed from selected IDs.
        expected: i128,
    },
    /// The forest differs from the stable `(weight, edge_id)` Kruskal result.
    NotStableMinimum {
        /// First missing canonical edge ID.
        missing_edge_id: String,
        /// First unexpected selected edge ID.
        unexpected_edge_id: String,
    },
    /// A non-tree edge is strictly better than a tree edge on its path.
    NonTreeEdgeImprovesForest {
        /// Non-tree edge that should replace `tree_edge_id`.
        non_tree_edge_id: String,
        /// Heavier or later-ID tree edge on the non-tree edge's path.
        tree_edge_id: String,
    },
}

/// Witness result. Failure details are separated from successful explanatory facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WitnessReport {
    /// True when every independent check passes.
    pub accepted: bool,
    /// Concrete failure categories; empty for an accepted publish.
    pub failures: Vec<WitnessFailure>,
    /// Independent canonical forest IDs.
    pub canonical_forest: BTreeSet<EdgeId>,
    /// Independently summed forest total.
    pub expected_total_weight: i128,
    /// Exact full-graph component count, including isolated vertices.
    pub graph_component_count: usize,
}

/// Verify a complete graph and its claimed forest IDs without using the core.
pub fn verify(
    graph: &Graph,
    claimed_forest: &BTreeSet<EdgeId>,
    reported_total: i128,
) -> WitnessReport {
    let mut failures = Vec::new();
    let canonical = canonical_kruskal(graph, &mut failures);
    let expected_total = sum_edges(graph, &canonical, &mut failures);
    let graph_components = graph_components(graph);

    check_present_non_loop(graph, claimed_forest, &mut failures);
    check_forest_shape(graph, claimed_forest, &graph_components, &mut failures);
    check_claimed_total(graph, claimed_forest, reported_total, &mut failures);
    check_canonical_set(claimed_forest, &canonical, &mut failures);
    check_non_tree_exchange(graph, claimed_forest, &mut failures);

    WitnessReport {
        accepted: failures.is_empty(),
        failures,
        canonical_forest: canonical,
        expected_total_weight: expected_total,
        graph_component_count: graph_components.len(),
    }
}

fn canonical_kruskal(graph: &Graph, failures: &mut Vec<WitnessFailure>) -> BTreeSet<EdgeId> {
    let mut parent: BTreeMap<Vertex, Vertex> = graph
        .vertices()
        .map(|vertex| (vertex.clone(), vertex.clone()))
        .collect();
    let mut selected = BTreeSet::new();

    let mut edges: Vec<&Edge> = graph.edges().filter(|edge| !edge.is_self_loop()).collect();
    edges.sort_by(|a, b| (a.weight, a.id.as_str()).cmp(&(b.weight, b.id.as_str())));

    for edge in edges {
        if independent_union(&mut parent, &edge.u, &edge.v) {
            selected.insert(edge.id.clone());
        }
    }

    for edge in graph.edges().filter(|edge| edge.is_self_loop()) {
        if selected.contains(&edge.id) {
            failures.push(WitnessFailure::SelfLoopSelected(edge.id.clone()));
        }
    }

    selected
}

fn check_present_non_loop(
    graph: &Graph,
    forest_ids: &BTreeSet<EdgeId>,
    failures: &mut Vec<WitnessFailure>,
) {
    for id in forest_ids {
        match graph.edge(id) {
            None => failures.push(WitnessFailure::MissingEdge(id.clone())),
            Some(edge) if edge.is_self_loop() => {
                failures.push(WitnessFailure::SelfLoopSelected(id.clone()));
            }
            Some(_) => {}
        }
    }

    for edge in graph.edges().filter(|edge| edge.is_self_loop()) {
        if !forest_ids.contains(&edge.id) {
            continue;
        }
        failures.push(WitnessFailure::SelfLoopNotExcluded(edge.id.clone()));
    }
}

fn check_forest_shape(
    graph: &Graph,
    forest_ids: &BTreeSet<EdgeId>,
    graph_components: &[BTreeSet<Vertex>],
    failures: &mut Vec<WitnessFailure>,
) {
    let usable: BTreeMap<EdgeId, &Edge> = forest_ids
        .iter()
        .filter_map(|id| graph.edge(id).map(|edge| (id.clone(), edge)))
        .filter(|(_, edge)| !edge.is_self_loop())
        .collect();

    for (edge_id, edge) in &usable {
        if connects_without_edge(&usable, edge_id, &edge.u, &edge.v) {
            failures.push(WitnessFailure::CycleDetected(edge.id.clone()));
            break;
        }
    }

    let forest = Forest::from_selected(usable.values().copied(), forest_ids);

    for component in graph_components {
        if component.len() <= 1 {
            continue;
        }
        let root = component
            .first()
            .cloned()
            .expect("non-empty component has a root");
        let reached = forest.component_from(&root);
        let reached_count = reached.intersection(&component).count();
        if reached_count != component.len() {
            failures.push(WitnessFailure::NotSpanning {
                component_root: root,
                graph_vertices: component.len(),
                forest_vertices: reached_count,
            });
        }

        for edge in forest_ids.iter().filter_map(|id| graph.edge(id)) {
            if component.contains(&edge.u) != component.contains(&edge.v) {
                failures.push(WitnessFailure::SpansDisconnectedComponents {
                    edge_id: edge.id.clone(),
                    u: edge.u.clone(),
                    v: edge.v.clone(),
                });
            }
        }
    }
}

fn check_claimed_total(
    graph: &Graph,
    forest_ids: &BTreeSet<EdgeId>,
    reported_total: i128,
    failures: &mut Vec<WitnessFailure>,
) {
    let summed: i128 = forest_ids
        .iter()
        .filter_map(|id| graph.edge(id))
        .filter(|edge| !edge.is_self_loop())
        .map(|edge| i128::from(edge.weight))
        .sum();

    if summed != reported_total {
        failures.push(WitnessFailure::TotalMismatch {
            reported: reported_total,
            expected: summed,
        });
    }
}

fn check_canonical_set(
    claimed: &BTreeSet<EdgeId>,
    canonical: &BTreeSet<EdgeId>,
    failures: &mut Vec<WitnessFailure>,
) {
    if claimed == canonical {
        return;
    }
    let missing = canonical
        .difference(claimed)
        .next()
        .cloned()
        .unwrap_or_default();
    let unexpected = claimed
        .difference(canonical)
        .next()
        .cloned()
        .unwrap_or_default();
    failures.push(WitnessFailure::NotStableMinimum {
        missing_edge_id: missing,
        unexpected_edge_id: unexpected,
    });
}

fn check_non_tree_exchange(
    graph: &Graph,
    forest_ids: &BTreeSet<EdgeId>,
    failures: &mut Vec<WitnessFailure>,
) {
    let usable: Vec<&Edge> = forest_ids
        .iter()
        .filter_map(|id| graph.edge(id))
        .filter(|edge| !edge.is_self_loop())
        .collect();
    let forest = Forest::from_selected(usable.into_iter(), forest_ids);

    for edge in graph
        .edges()
        .filter(|edge| !edge.is_self_loop() && !forest_ids.contains(&edge.id))
    {
        let Some(path) = forest.path(&edge.u, &edge.v) else {
            continue;
        };
        for tree_id in path {
            let Some(tree_edge) = graph.edge(&tree_id) else {
                continue;
            };
            if (edge.weight, edge.id.as_str()) < (tree_edge.weight, tree_edge.id.as_str()) {
                failures.push(WitnessFailure::NonTreeEdgeImprovesForest {
                    non_tree_edge_id: edge.id.clone(),
                    tree_edge_id: tree_edge.id.clone(),
                });
                return;
            }
        }
    }
}

fn sum_edges(
    graph: &Graph,
    edge_ids: &BTreeSet<EdgeId>,
    _failures: &mut Vec<WitnessFailure>,
) -> i128 {
    edge_ids
        .iter()
        .filter_map(|id| graph.edge(id))
        .map(|edge| i128::from(edge.weight))
        .sum()
}

fn graph_components(graph: &Graph) -> Vec<BTreeSet<Vertex>> {
    let mut adjacency: BTreeMap<Vertex, BTreeSet<Vertex>> = graph
        .vertices()
        .map(|vertex| (vertex.clone(), BTreeSet::new()))
        .collect();
    for edge in graph.edges().filter(|edge| !edge.is_self_loop()) {
        adjacency
            .entry(edge.u.clone())
            .or_default()
            .insert(edge.v.clone());
        adjacency
            .entry(edge.v.clone())
            .or_default()
            .insert(edge.u.clone());
    }

    let mut unseen: BTreeSet<Vertex> = adjacency.keys().cloned().collect();
    let mut components = Vec::new();

    while let Some(root) = unseen.first().cloned() {
        let mut component = BTreeSet::new();
        let mut queue = VecDeque::from([root.clone()]);
        component.insert(root.clone());
        unseen.remove(&root);

        while let Some(vertex) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&vertex) {
                for neighbor in neighbors {
                    if component.insert(neighbor.clone()) {
                        unseen.remove(neighbor);
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        components.push(component);
    }

    components
}

fn connects_without_edge(
    edges: &BTreeMap<EdgeId, &Edge>,
    excluded_edge_id: &str,
    source: &str,
    target: &str,
) -> bool {
    let mut adjacency: BTreeMap<Vertex, Vec<&Vertex>> = BTreeMap::new();
    for (edge_id, edge) in edges {
        if edge_id == excluded_edge_id {
            continue;
        }
        adjacency.entry(edge.u.clone()).or_default().push(&edge.v);
        adjacency.entry(edge.v.clone()).or_default().push(&edge.u);
    }

    let mut seen = BTreeSet::from([source.to_string()]);
    let mut queue = VecDeque::from([source.to_string()]);
    while let Some(vertex) = queue.pop_front() {
        if vertex == target {
            return true;
        }
        if let Some(neighbors) = adjacency.get(&vertex) {
            for neighbor in neighbors {
                if seen.insert((*neighbor).clone()) {
                    queue.push_back((*neighbor).clone());
                }
            }
        }
    }
    false
}

fn independent_find(parent: &mut BTreeMap<Vertex, Vertex>, vertex: &Vertex) -> Vertex {
    let mut current = vertex.clone();
    while let Some(next) = parent.get(&current) {
        if *next == current {
            return current;
        }
        current = next.clone();
    }
    current
}

fn independent_union(parent: &mut BTreeMap<Vertex, Vertex>, left: &str, right: &str) -> bool {
    let left_root = independent_find(parent, &left.to_string());
    let right_root = independent_find(parent, &right.to_string());
    if left_root == right_root {
        return false;
    }
    parent.insert(right_root.clone(), left_root.clone());
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Weight;

    fn e(id: &str, u: &str, v: &str, weight: Weight) -> Edge {
        Edge {
            id: id.to_string(),
            u: u.to_string(),
            v: v.to_string(),
            weight,
        }
    }

    #[test]
    fn accepts_canonical_disconnected_forest() {
        let mut graph = Graph::new();
        graph.upsert_edge(e("a1", "a", "b", 1));
        graph.upsert_edge(e("a2", "b", "c", 2));
        graph.upsert_edge(e("x1", "x", "y", 3));
        graph.upsert_edge(e("loop", "a", "a", -9));

        let selected = BTreeSet::from(["a1".to_string(), "a2".to_string(), "x1".to_string()]);
        let report = verify(&graph, &selected, 6);

        assert!(
            report.accepted,
            "unexpected failures: {:?}",
            report.failures
        );
        assert_eq!(report.graph_component_count, 2);
    }

    #[test]
    fn reports_non_tree_improvement() {
        let mut graph = Graph::new();
        graph.upsert_edge(e("heavy", "a", "b", 5));
        graph.upsert_edge(e("cheap", "a", "b", 1));

        let selected = BTreeSet::from(["heavy".to_string()]);
        let report = verify(&graph, &selected, 5);

        assert!(!report.accepted);
        assert!(report
            .failures
            .iter()
            .any(|failure| matches!(failure, WitnessFailure::NotStableMinimum { .. })));
    }
}
