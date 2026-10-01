//! Restore concrete paths from a matching in the split-vertex bipartite graph.

use serde::Serialize;

use crate::matching::MatchingOutcome;
use crate::model::{Graph, Vertex};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PathCover {
    pub paths: Vec<Vec<usize>>,
    pub path_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RestoreError {
    InvalidMatchingSize {
        matching_size: usize,
        vertex_count: usize,
    },
    LeftVertexReused {
        vertex: usize,
    },
    RightVertexReused {
        vertex: usize,
    },
    MatchingEdgeNotInDag {
        from: usize,
        to: usize,
    },
    VertexRepeatedInRestoredPath {
        vertex: usize,
    },
    RestoredVertexCountMismatch {
        covered: usize,
        vertex_count: usize,
    },
    RestoredVertexMissing {
        vertex: usize,
    },
}

/// Restores vertex-disjoint paths from matching edges.
///
/// A matching edge `u -> v` means `u` is immediately followed by `v` in a path.
/// Path starts are right-copy vertices that no left-copy vertex matched into.
pub fn restore_path_cover(
    graph: &Graph,
    matching: &MatchingOutcome,
) -> Result<PathCover, RestoreError> {
    let n = graph.vertex_count();
    if matching.size > n {
        return Err(RestoreError::InvalidMatchingSize {
            matching_size: matching.size,
            vertex_count: n,
        });
    }

    let mut next_vertex = vec![None; n];
    let mut previous_vertex = vec![None; n];

    for edge in &matching.edges {
        if !graph.has_edge(edge.from, edge.to) {
            return Err(RestoreError::MatchingEdgeNotInDag {
                from: edge.from,
                to: edge.to,
            });
        }
        if next_vertex[edge.from].is_some() {
            return Err(RestoreError::LeftVertexReused { vertex: edge.from });
        }
        if previous_vertex[edge.to].is_some() {
            return Err(RestoreError::RightVertexReused { vertex: edge.to });
        }
        next_vertex[edge.from] = Some(edge.to);
        previous_vertex[edge.to] = Some(edge.from);
    }

    let mut paths = Vec::new();

    for start in 0..n {
        if previous_vertex[start].is_some() {
            continue;
        }

        let mut path = Vec::new();
        let mut current: Option<Vertex> = Some(start);

        while let Some(vertex) = current {
            if path.contains(&vertex) {
                return Err(RestoreError::VertexRepeatedInRestoredPath { vertex });
            }
            path.push(vertex);
            current = next_vertex[vertex];
        }

        paths.push(path);
    }

    let mut seen = vec![false; n];
    for path in &paths {
        for &vertex in path {
            if vertex >= n {
                return Err(RestoreError::RestoredVertexMissing { vertex });
            }
            if seen[vertex] {
                return Err(RestoreError::VertexRepeatedInRestoredPath { vertex });
            }
            seen[vertex] = true;
        }
    }

    let covered = seen.iter().filter(|visited| **visited).count();
    if covered != n {
        if let Some(vertex) = seen.iter().position(|visited| !*visited) {
            return Err(RestoreError::RestoredVertexMissing { vertex });
        }
        return Err(RestoreError::RestoredVertexCountMismatch {
            covered,
            vertex_count: n,
        });
    }

    paths.sort_by_key(|path| *path.first().expect("a restored path is nonempty"));
    let path_count = paths.len();

    Ok(PathCover { paths, path_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::{MatchingOutcome, MatchingPhase};
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

    fn matching(edges: &[(usize, usize)]) -> MatchingOutcome {
        MatchingOutcome {
            edges: edges.iter().map(|&(from, to)| Edge { from, to }).collect(),
            size: edges.len(),
            phases: vec![MatchingPhase {
                phase: 1,
                augmenting_paths: edges.len(),
            }],
        }
    }

    #[test]
    fn zero_edge_graph_restores_all_singletons() {
        let graph = graph(3, &[]);
        let cover = restore_path_cover(&graph, &matching(&[])).expect("singleton cover");

        assert_eq!(cover.path_count, 3);
        assert_eq!(cover.paths, vec![vec![0], vec![1], vec![2]]);
        assert_eq!(cover.path_count + 0, graph.vertex_count());
    }

    #[test]
    fn restores_diamond_chains() {
        let graph = graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let cover = restore_path_cover(&graph, &matching(&[(0, 1), (1, 3)])).expect("path cover");

        assert_eq!(cover.path_count, 2);
        assert_eq!(cover.paths, vec![vec![0, 1, 3], vec![2]]);
    }

    #[test]
    fn rejects_reused_right_vertex() {
        let graph = graph(4, &[(0, 2), (1, 2)]);
        let err = restore_path_cover(&graph, &matching(&[(0, 2), (1, 2)]))
            .expect_err("matching must be one-to-one");

        assert_eq!(err, RestoreError::RightVertexReused { vertex: 2 });
    }
}
