//! Hopcroft-Karp maximum bipartite matching for the DAG path-cover reduction.

use serde::Serialize;

use crate::model::{Edge, Graph, Vertex};

const INF: usize = usize::MAX;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MatchingPhase {
    pub phase: usize,
    pub augmenting_paths: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MatchingOutcome {
    pub edges: Vec<Edge>,
    pub size: usize,
    pub phases: Vec<MatchingPhase>,
}

/// Computes a maximum matching in the bipartite graph formed from each directed
/// edge: left-copy `u` can match right-copy `v`.
pub fn hopcroft_karp(graph: &Graph) -> MatchingOutcome {
    let n = graph.vertex_count();
    let mut pair_left = vec![None; n];
    let mut pair_right = vec![None; n];
    let mut dist = vec![INF; n];
    let mut phases = Vec::new();
    let mut phase_number = 0;

    while bfs(graph, &pair_left, &pair_right, &mut dist) {
        phase_number += 1;
        let mut augmenting_paths = 0;

        for u in 0..n {
            if pair_left[u].is_none() && dfs(u, graph, &mut pair_left, &mut pair_right, &mut dist) {
                augmenting_paths += 1;
            }
        }

        phases.push(MatchingPhase {
            phase: phase_number,
            augmenting_paths,
        });
    }

    let edges: Vec<Edge> = pair_left
        .iter()
        .enumerate()
        .filter_map(|(from, matched_to)| matched_to.map(|to| Edge { from, to }))
        .collect();
    let size = edges.len();

    MatchingOutcome {
        edges,
        size,
        phases,
    }
}

fn bfs(
    graph: &Graph,
    pair_left: &[Option<Vertex>],
    pair_right: &[Option<Vertex>],
    dist: &mut [usize],
) -> bool {
    let mut queue = std::collections::VecDeque::new();

    for u in 0..graph.vertex_count() {
        if pair_left[u].is_none() {
            dist[u] = 0;
            queue.push_back(u);
        } else {
            dist[u] = INF;
        }
    }

    let mut reaches_free_right_vertex = false;

    while let Some(u) = queue.pop_front() {
        for &v in graph.neighbors(u) {
            match pair_right[v] {
                None => reaches_free_right_vertex = true,
                Some(next_u) if dist[next_u] == INF => {
                    dist[next_u] = dist[u] + 1;
                    queue.push_back(next_u);
                }
                Some(_) => {}
            }
        }
    }

    reaches_free_right_vertex
}

fn dfs(
    u: Vertex,
    graph: &Graph,
    pair_left: &mut [Option<Vertex>],
    pair_right: &mut [Option<Vertex>],
    dist: &mut [usize],
) -> bool {
    for &v in graph.neighbors(u) {
        let can_augment = match pair_right[v] {
            None => true,
            Some(next_u) => {
                dist[next_u] == dist[u] + 1 && dfs(next_u, graph, pair_left, pair_right, dist)
            }
        };

        if can_augment {
            pair_left[u] = Some(v);
            pair_right[v] = Some(u);
            return true;
        }
    }

    dist[u] = INF;
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Edge, GraphInput};

    fn graph(n: usize, edges: &[(usize, usize)]) -> Graph {
        Graph::new(
            &GraphInput {
                vertices: n,
                edges: edges.iter().map(|&(from, to)| Edge { from, to }).collect(),
            },
            100,
        )
        .expect("test graph")
    }

    #[test]
    fn zero_edge_graph_has_empty_matching() {
        let outcome = hopcroft_karp(&graph(3, &[]));
        assert_eq!(outcome.size, 0);
        assert!(outcome.edges.is_empty());
    }

    #[test]
    fn diamond_has_matching_size_two() {
        let outcome = hopcroft_karp(&graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]));

        assert_eq!(outcome.size, 2);
        let left: Vec<usize> = outcome.edges.iter().map(|edge| edge.from).collect();
        let right: Vec<usize> = outcome.edges.iter().map(|edge| edge.to).collect();
        assert_eq!(
            left.len(),
            left.iter().collect::<std::collections::HashSet<_>>().len()
        );
        assert_eq!(
            right.len(),
            right.iter().collect::<std::collections::HashSet<_>>().len()
        );
    }
}
