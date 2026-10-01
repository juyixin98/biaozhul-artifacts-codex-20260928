//! Independent verification of a path-cover witness and matching relation.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::matching::MatchingOutcome;
use crate::model::{Edge, Graph};
use crate::path_cover::PathCover;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WitnessProblem {
    EmptyPath {
        path_index: usize,
    },
    VertexOutOfBounds {
        vertex: usize,
        vertex_count: usize,
    },
    VertexMissing {
        vertex: usize,
    },
    VertexRepeated {
        vertex: usize,
    },
    PathEdgeNotInDag {
        from: usize,
        to: usize,
    },
    MatchingEdgeNotInDag {
        from: usize,
        to: usize,
    },
    MatchingLeftReused {
        vertex: usize,
    },
    MatchingRightReused {
        vertex: usize,
    },
    MatchingSizeMismatch {
        declared: usize,
        actual: usize,
    },
    PathTransitionNotMatched {
        from: usize,
        to: usize,
    },
    MatchingEdgeNotUsedByPaths {
        from: usize,
        to: usize,
    },
    CountRelationInvalid {
        vertex_count: usize,
        matching_size: usize,
        path_count: usize,
    },
}

impl WitnessProblem {
    pub fn code(&self) -> &'static str {
        match self {
            Self::EmptyPath { .. } => "empty_path",
            Self::VertexOutOfBounds { .. } => "vertex_out_of_bounds",
            Self::VertexMissing { .. } => "vertex_missing",
            Self::VertexRepeated { .. } => "vertex_repeated",
            Self::PathEdgeNotInDag { .. } => "path_edge_not_in_dag",
            Self::MatchingEdgeNotInDag { .. } => "matching_edge_not_in_dag",
            Self::MatchingLeftReused { .. } => "matching_left_reused",
            Self::MatchingRightReused { .. } => "matching_right_reused",
            Self::MatchingSizeMismatch { .. } => "matching_size_mismatch",
            Self::PathTransitionNotMatched { .. } => "path_transition_not_matched",
            Self::MatchingEdgeNotUsedByPaths { .. } => "matching_edge_not_used_by_paths",
            Self::CountRelationInvalid { .. } => "count_relation_invalid",
        }
    }

    pub fn explanation(&self) -> &'static str {
        match self {
            Self::EmptyPath { .. } => "a path must contain at least one vertex",
            Self::VertexOutOfBounds { .. } => "a path refers to a vertex outside the graph",
            Self::VertexMissing { .. } => "every graph vertex must appear in exactly one path",
            Self::VertexRepeated { .. } => "a vertex appears more than once in the path partition",
            Self::PathEdgeNotInDag { .. } => "every consecutive pair in a path must be a directed DAG edge",
            Self::MatchingEdgeNotInDag { .. } => "each matching edge must correspond to an original directed edge",
            Self::MatchingLeftReused { .. } => "each left vertex copy may appear in at most one matching edge",
            Self::MatchingRightReused { .. } => "each right vertex copy may appear in at most one matching edge",
            Self::MatchingSizeMismatch { .. } => "the declared matching size does not equal its edge count",
            Self::PathTransitionNotMatched { .. } => "every internal path edge must be represented by the matching",
            Self::MatchingEdgeNotUsedByPaths { .. } => "every matching edge must be used as an internal path edge",
            Self::CountRelationInvalid { .. } => "a valid vertex-disjoint cover must satisfy path_count + matching_size = vertex_count",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WitnessReport {
    pub accepted: bool,
    pub vertex_count: usize,
    pub path_count: usize,
    pub matching_size: usize,
    pub path_transition_count: usize,
    pub checks: Vec<String>,
    pub problem: Option<WitnessProblem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WitnessRejection {
    pub code: String,
    pub message: String,
    pub details: WitnessProblem,
}

pub fn verify_witness(
    graph: &Graph,
    cover: &PathCover,
    matching: &MatchingOutcome,
) -> Result<WitnessReport, WitnessRejection> {
    let n = graph.vertex_count();
    let checks = vec![
        "every path is nonempty".to_string(),
        "every vertex id is in 0..n".to_string(),
        "vertices have no repetition or omission".to_string(),
        "consecutive path vertices are original directed edges".to_string(),
        "matching edges are original edges and one-to-one".to_string(),
        "matching edges are exactly the internal path edges".to_string(),
        "path_count + maximum_matching_size = vertex_count".to_string(),
    ];

    let reject = |problem: WitnessProblem| WitnessRejection {
        code: problem.code().to_string(),
        message: problem.explanation().to_string(),
        details: problem,
    };

    for (path_index, path) in cover.paths.iter().enumerate() {
        if path.is_empty() {
            return Err(reject(WitnessProblem::EmptyPath { path_index }));
        }
    }

    for path in &cover.paths {
        for &vertex in path {
            if vertex >= n {
                return Err(reject(WitnessProblem::VertexOutOfBounds {
                    vertex,
                    vertex_count: n,
                }));
            }
        }
    }

    let mut seen = vec![false; n];
    for path in &cover.paths {
        for &vertex in path {
            if seen[vertex] {
                return Err(reject(WitnessProblem::VertexRepeated { vertex }));
            }
            seen[vertex] = true;
        }
    }

    if let Some(vertex) = seen.iter().position(|visited| !*visited) {
        return Err(reject(WitnessProblem::VertexMissing { vertex }));
    }

    let mut transitions = HashSet::new();
    for path in &cover.paths {
        for pair in path.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            if !graph.has_edge(from, to) {
                return Err(reject(WitnessProblem::PathEdgeNotInDag { from, to }));
            }
            transitions.insert(Edge { from, to });
        }
    }

    if matching.edges.len() != matching.size {
        return Err(reject(WitnessProblem::MatchingSizeMismatch {
            declared: matching.size,
            actual: matching.edges.len(),
        }));
    }

    let mut matched_left = HashMap::new();
    let mut matched_right = HashMap::new();
    let mut matching_edges = HashSet::new();

    for edge in &matching.edges {
        if !graph.has_edge(edge.from, edge.to) {
            return Err(reject(WitnessProblem::MatchingEdgeNotInDag {
                from: edge.from,
                to: edge.to,
            }));
        }
        if let Some(_existing) = matched_left.insert(edge.from, edge.to) {
            return Err(reject(WitnessProblem::MatchingLeftReused {
                vertex: edge.from,
            }));
        }
        if let Some(_existing) = matched_right.insert(edge.to, edge.from) {
            return Err(reject(WitnessProblem::MatchingRightReused {
                vertex: edge.to,
            }));
        }
        matching_edges.insert(*edge);
    }

    for edge in &transitions {
        if !matching_edges.contains(edge) {
            return Err(reject(WitnessProblem::PathTransitionNotMatched {
                from: edge.from,
                to: edge.to,
            }));
        }
    }

    for edge in &matching_edges {
        if !transitions.contains(edge) {
            return Err(reject(WitnessProblem::MatchingEdgeNotUsedByPaths {
                from: edge.from,
                to: edge.to,
            }));
        }
    }

    if cover.path_count + matching.size != n || cover.path_count != cover.paths.len() {
        return Err(reject(WitnessProblem::CountRelationInvalid {
            vertex_count: n,
            matching_size: matching.size,
            path_count: cover.path_count,
        }));
    }

    Ok(WitnessReport {
        accepted: true,
        vertex_count: n,
        path_count: cover.path_count,
        matching_size: matching.size,
        path_transition_count: transitions.len(),
        checks,
        problem: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::{MatchingOutcome, MatchingPhase};
    use crate::model::GraphInput;
    use crate::path_cover::PathCover;

    fn graph(n: usize, edges: &[(usize, usize)]) -> Graph {
        Graph::new(
            &GraphInput {
                vertices: n,
                edges: edges
                    .iter()
                    .map(|edge| Edge {
                        from: edge.0,
                        to: edge.1,
                    })
                    .collect(),
            },
            100,
        )
        .expect("test graph")
    }

    fn matching(edges: &[(usize, usize)]) -> MatchingOutcome {
        let edges: Vec<Edge> = edges.iter().map(|&(from, to)| Edge { from, to }).collect();
        let size = edges.len();
        MatchingOutcome {
            edges,
            size,
            phases: vec![MatchingPhase {
                phase: 1,
                augmenting_paths: size,
            }],
        }
    }

    #[test]
    fn accepts_valid_witness_and_checks_formula() {
        let graph = graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let cover = PathCover {
            paths: vec![vec![0, 1, 3], vec![2]],
            path_count: 2,
        };
        let matching = matching(&[(0, 1), (1, 3)]);

        let report = verify_witness(&graph, &cover, &matching).expect("valid witness");
        assert!(report.accepted);
        assert_eq!(report.vertex_count, 4);
        assert_eq!(report.path_transition_count, 2);
        assert_eq!(
            report.path_count + report.matching_size,
            report.vertex_count
        );
    }

    #[test]
    fn rejects_missing_vertex_with_specific_category() {
        let graph = graph(3, &[(0, 1)]);
        let cover = PathCover {
            paths: vec![vec![0, 1]],
            path_count: 1,
        };
        let matching = matching(&[(0, 1)]);

        let rejection = verify_witness(&graph, &cover, &matching).expect_err("vertex 2 missing");
        assert_eq!(rejection.code, "vertex_missing");
        assert_eq!(
            rejection.details,
            WitnessProblem::VertexMissing { vertex: 2 }
        );
    }

    #[test]
    fn rejects_unused_matching_edge_with_specific_category() {
        let graph = graph(4, &[(0, 1)]);
        let cover = PathCover {
            paths: vec![vec![0], vec![1], vec![2], vec![3]],
            path_count: 4,
        };
        let matching = matching(&[(0, 1)]);

        let rejection = verify_witness(&graph, &cover, &matching).expect_err("extra edge");
        assert_eq!(rejection.code, "matching_edge_not_used_by_paths");
        assert_eq!(
            rejection.details,
            WitnessProblem::MatchingEdgeNotUsedByPaths { from: 0, to: 1 }
        );
    }

    #[test]
    fn rejects_broken_count_relation_with_specific_category() {
        let graph = graph(3, &[(0, 1), (1, 2)]);
        let cover = PathCover {
            paths: vec![vec![0, 1, 2]],
            path_count: 1,
        };
        let empty_matching = matching(&[]);

        let transition_problem = verify_witness(&graph, &cover, &empty_matching)
            .expect_err("internal path transitions require matching edges")
            .code;
        assert_eq!(transition_problem, "path_transition_not_matched");

        let cover_with_wrong_metadata = PathCover {
            paths: vec![vec![0, 1], vec![2]],
            path_count: 1,
        };
        let one_edge_matching = matching(&[(0, 1)]);
        let metadata_problem =
            verify_witness(&graph, &cover_with_wrong_metadata, &one_edge_matching)
                .expect_err("metadata path_count disagrees with actual paths")
                .code;
        assert_eq!(metadata_problem, "count_relation_invalid");
    }
}
