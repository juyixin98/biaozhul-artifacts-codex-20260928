use std::collections::{HashMap, HashSet};

use crate::algorithm::PathCover;
use crate::model::Graph;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessFailure {
    EmptyPath,
    VertexOutOfRange {
        vertex: usize,
    },
    RepeatedVertex {
        vertex: usize,
    },
    MissingVertex {
        vertex: usize,
    },
    UnknownPathEdge {
        from: usize,
        to: usize,
    },
    EmptyMatchingPair,
    MatchingPairNotAnEdge {
        from: usize,
        to: usize,
    },
    MatchingLeftReused {
        left: usize,
    },
    MatchingRightReused {
        right: usize,
    },
    PathCountMismatch {
        claimed: usize,
        actual: usize,
    },
    MatchingSizeMismatch {
        claimed: usize,
        actual: usize,
    },
    CardinalityRelationFailed {
        vertex_count: usize,
        matching_size: usize,
        path_count: usize,
    },
    PathAndMatchingMismatch {
        from: usize,
        to: Option<usize>,
    },
}

impl WitnessFailure {
    pub fn category(&self) -> &'static str {
        match self {
            WitnessFailure::EmptyPath => "EMPTY_PATH",
            WitnessFailure::VertexOutOfRange { .. } => "VERTEX_OUT_OF_RANGE",
            WitnessFailure::RepeatedVertex { .. } => "REPEATED_VERTEX",
            WitnessFailure::MissingVertex { .. } => "MISSING_VERTEX",
            WitnessFailure::UnknownPathEdge { .. } => "UNKNOWN_PATH_EDGE",
            WitnessFailure::EmptyMatchingPair => "EMPTY_MATCHING_PAIR",
            WitnessFailure::MatchingPairNotAnEdge { .. } => "MATCHING_PAIR_NOT_AN_EDGE",
            WitnessFailure::MatchingLeftReused { .. } => "MATCHING_LEFT_REUSED",
            WitnessFailure::MatchingRightReused { .. } => "MATCHING_RIGHT_REUSED",
            WitnessFailure::PathCountMismatch { .. } => "PATH_COUNT_MISMATCH",
            WitnessFailure::MatchingSizeMismatch { .. } => "MATCHING_SIZE_MISMATCH",
            WitnessFailure::CardinalityRelationFailed { .. } => "CARDINALITY_RELATION_FAILED",
            WitnessFailure::PathAndMatchingMismatch { .. } => "PATH_AND_MATCHING_MISMATCH",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessCheck {
    pub valid: bool,
    pub failures: Vec<WitnessFailure>,
    pub vertex_count: usize,
    pub path_count: usize,
    pub matching_size: usize,
    pub basis: Vec<String>,
}

pub fn verify_path_cover(graph: &Graph, claimed: &PathCover) -> WitnessCheck {
    let mut failures = Vec::new();
    let edge_set: HashSet<(usize, usize)> = graph.edges.iter().copied().collect();
    let mut covered = HashSet::new();

    for path in &claimed.paths {
        if path.is_empty() {
            failures.push(WitnessFailure::EmptyPath);
            continue;
        }
        for &vertex in path {
            if vertex >= graph.vertex_count {
                failures.push(WitnessFailure::VertexOutOfRange { vertex });
                continue;
            }
            if !covered.insert(vertex) {
                failures.push(WitnessFailure::RepeatedVertex { vertex });
            }
        }
        for edge in path.windows(2) {
            let (from, to) = (edge[0], edge[1]);
            if from < graph.vertex_count
                && to < graph.vertex_count
                && !edge_set.contains(&(from, to))
            {
                failures.push(WitnessFailure::UnknownPathEdge { from, to });
            }
        }
    }

    for vertex in 0..graph.vertex_count {
        if !covered.contains(&vertex) {
            failures.push(WitnessFailure::MissingVertex { vertex });
        }
    }

    if claimed.paths.len() != claimed.minimum_path_count {
        failures.push(WitnessFailure::PathCountMismatch {
            claimed: claimed.minimum_path_count,
            actual: claimed.paths.len(),
        });
    }

    if claimed.matching.pairs.len() != claimed.matching.size {
        failures.push(WitnessFailure::MatchingSizeMismatch {
            claimed: claimed.matching.size,
            actual: claimed.matching.pairs.len(),
        });
    }

    let mut matching_edges = HashSet::new();
    let mut left_seen = HashSet::new();
    let mut right_seen = HashSet::new();
    for &(from, to) in &claimed.matching.pairs {
        if from == to {
            failures.push(WitnessFailure::EmptyMatchingPair);
        }
        if from >= graph.vertex_count || to >= graph.vertex_count {
            failures.push(WitnessFailure::VertexOutOfRange {
                vertex: from.max(to),
            });
        } else if !edge_set.contains(&(from, to)) {
            failures.push(WitnessFailure::MatchingPairNotAnEdge { from, to });
        }
        if !left_seen.insert(from) {
            failures.push(WitnessFailure::MatchingLeftReused { left: from });
        }
        if !right_seen.insert(to) {
            failures.push(WitnessFailure::MatchingRightReused { right: to });
        }
        matching_edges.insert((from, to));
    }

    let independent_path_count = claimed.paths.len();
    let expected_path_count = graph.vertex_count.checked_sub(claimed.matching.size);
    if expected_path_count != Some(independent_path_count) {
        failures.push(WitnessFailure::CardinalityRelationFailed {
            vertex_count: graph.vertex_count,
            matching_size: claimed.matching.size,
            path_count: independent_path_count,
        });
    }

    let unique_matching_edges = claimed.matching.pairs.iter().collect::<HashSet<_>>().len();
    if unique_matching_edges != claimed.matching.size {
        failures.push(WitnessFailure::MatchingSizeMismatch {
            claimed: claimed.matching.size,
            actual: unique_matching_edges,
        });
    }

    let mut path_successors = HashMap::new();
    for path in &claimed.paths {
        for edge in path.windows(2) {
            path_successors.insert(edge[0], edge[1]);
        }
    }

    for &(from, to) in &matching_edges {
        if path_successors.get(&from) != Some(&to) {
            failures.push(WitnessFailure::PathAndMatchingMismatch {
                from,
                to: path_successors.get(&from).copied(),
            });
        }
    }
    for (&from, &to) in &path_successors {
        if !matching_edges.contains(&(from, to)) {
            failures.push(WitnessFailure::PathAndMatchingMismatch { from, to: Some(to) });
        }
    }

    let path_count = claimed.paths.len();
    let matching_size = claimed.matching.size;
    let basis = vec![
        format!(
            "covered {}/{} vertices exactly once",
            covered.len(),
            graph.vertex_count
        ),
        format!("restored {} path(s)", path_count),
        format!("bipartite matching size = {}", matching_size),
        format!(
            "|V| - |M| = {} - {} = {}",
            graph.vertex_count,
            matching_size,
            graph.vertex_count - matching_size
        ),
    ];

    WitnessCheck {
        valid: failures.is_empty(),
        failures,
        vertex_count: graph.vertex_count,
        path_count,
        matching_size,
        basis,
    }
}
