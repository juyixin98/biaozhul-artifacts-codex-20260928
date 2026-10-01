//! Algorithm orchestration: validate, reject cycles, match, restore, verify.

use serde::Serialize;

use crate::cycle::find_directed_cycle;
use crate::matching::{hopcroft_karp, MatchingOutcome};
use crate::model::{Graph, GraphBuildError, GraphInput};
use crate::path_cover::{restore_path_cover, PathCover, RestoreError};
use crate::witness::{verify_witness, WitnessRejection, WitnessReport};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GraphSummary {
    pub vertices: usize,
    pub input_edges: usize,
    pub distinct_edges: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ComputeTrace {
    pub steps: Vec<String>,
    pub decision_basis: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SolveResponse {
    pub run_id: String,
    pub version: String,
    pub graph: GraphSummary,
    pub matching: MatchingOutcome,
    pub paths: Vec<Vec<usize>>,
    pub path_count: usize,
    pub minimum_path_count: usize,
    pub witness: WitnessReport,
    pub trace: ComputeTrace,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SolveError {
    InvalidGraph {
        details: GraphBuildError,
    },
    CycleDetected {
        code: &'static str,
        message: &'static str,
        evidence: crate::cycle::CycleEvidence,
    },
    RestoreFailed {
        details: RestoreError,
    },
    WitnessRejected {
        details: WitnessRejection,
    },
}

impl SolveError {
    pub fn status_code(&self) -> u16 {
        match self {
            Self::InvalidGraph { .. } => 400,
            Self::CycleDetected { .. } => 422,
            Self::RestoreFailed { .. } | Self::WitnessRejected { .. } => 500,
        }
    }

    pub fn code(&self) -> String {
        match self {
            Self::InvalidGraph { details } => match details {
                GraphBuildError::VertexCountOutOfRange { .. } => {
                    "invalid_vertex_count_out_of_range".to_string()
                }
                GraphBuildError::EdgeOutOfBounds { .. } => "invalid_edge_out_of_bounds".to_string(),
            },
            Self::CycleDetected { code, .. } => (*code).to_string(),
            Self::RestoreFailed { details } => match details {
                RestoreError::InvalidMatchingSize { .. } => {
                    "restore_invalid_matching_size".to_string()
                }
                RestoreError::LeftVertexReused { .. } => "restore_left_vertex_reused".to_string(),
                RestoreError::RightVertexReused { .. } => "restore_right_vertex_reused".to_string(),
                RestoreError::MatchingEdgeNotInDag { .. } => "restore_edge_not_in_dag".to_string(),
                RestoreError::VertexRepeatedInRestoredPath { .. } => {
                    "restore_vertex_repeated".to_string()
                }
                RestoreError::RestoredVertexCountMismatch { .. } => {
                    "restore_vertex_count_mismatch".to_string()
                }
                RestoreError::RestoredVertexMissing { .. } => "restore_vertex_missing".to_string(),
            },
            Self::WitnessRejected { details } => details.code.clone(),
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::InvalidGraph { details } => match details {
                GraphBuildError::VertexCountOutOfRange {
                    actual,
                    max_vertices,
                } => {
                    format!("vertex count {actual} exceeds configured limit {max_vertices}")
                }
                GraphBuildError::EdgeOutOfBounds { edge, vertices } => {
                    format!(
                        "edge {} -> {} is outside a graph with {vertices} vertices",
                        edge.from, edge.to
                    )
                }
            },
            Self::CycleDetected { message, .. } => (*message).to_string(),
            Self::RestoreFailed { details } => serde_json::to_string(details)
                .unwrap_or_else(|_| "path restoration failed".to_string()),
            Self::WitnessRejected { details } => details.message.clone(),
        }
    }
}

pub fn solve_dag_path_cover(
    run_id: &str,
    input: &GraphInput,
    max_vertices: usize,
) -> Result<SolveResponse, SolveError> {
    tracing::info!(
        run_id = %run_id,
        vertices = input.vertices,
        input_edges = input.edges.len(),
        "validate input graph"
    );

    let graph: Graph = Graph::new(input, max_vertices).map_err(|details| {
        tracing::error!(run_id = %run_id, ?details, "graph validation failed");
        SolveError::InvalidGraph { details }
    })?;

    let summary = GraphSummary {
        vertices: graph.vertex_count(),
        input_edges: graph.input_edge_count(),
        distinct_edges: graph.distinct_edge_count(),
    };

    let mut trace = ComputeTrace {
        steps: vec![
            format!(
                "validated dense vertex ids 0..{} and {} input edges ({} distinct)",
                summary.vertices, summary.input_edges, summary.distinct_edges
            ),
            "running iterative directed DFS cycle detection".to_string(),
        ],
        decision_basis: vec![
            "directed-cycle evidence is required before applying the DAG theorem".to_string(),
        ],
    };

    if let Some(evidence) = find_directed_cycle(&graph) {
        tracing::error!(run_id = %run_id, ?evidence, "directed cycle detected");
        return Err(SolveError::CycleDetected {
            code: "cycle_detected",
            message: "input graph contains a directed cycle; minimum DAG path cover is not defined",
            evidence,
        });
    }

    trace
        .steps
        .push("cycle check found no directed cycle".to_string());
    trace
        .steps
        .push("constructing split-vertex bipartite graph from directed edges".to_string());

    let matching: MatchingOutcome = hopcroft_karp(&graph);
    tracing::info!(
        run_id = %run_id,
        matching_size = matching.size,
        phases = matching.phases.len(),
        "Hopcroft-Karp maximum matching complete"
    );
    for phase in &matching.phases {
        tracing::debug!(
            run_id = %run_id,
            phase = phase.phase,
            augmenting_paths = phase.augmenting_paths,
            "matching phase"
        );
    }
    trace.steps.push(format!(
        "Hopcroft-Karp returned maximum matching size {} in {} phase(s)",
        matching.size,
        matching.phases.len()
    ));

    let cover: PathCover = restore_path_cover(&graph, &matching).map_err(|details| {
        tracing::error!(run_id = %run_id, ?details, "path restoration failed");
        SolveError::RestoreFailed { details }
    })?;

    if cover.path_count + matching.size != graph.vertex_count() {
        tracing::error!(
            run_id = %run_id,
            path_count = cover.path_count,
            matching_size = matching.size,
            vertex_count = graph.vertex_count(),
            "independent count relation failed"
        );
    }

    trace.steps.push(format!(
        "restored {} path(s); relation {} + {} = {} checked",
        cover.path_count,
        cover.path_count,
        matching.size,
        graph.vertex_count()
    ));

    let witness = verify_witness(&graph, &cover, &matching).map_err(|details| {
        tracing::error!(run_id = %run_id, ?details, "witness verification rejected result");
        SolveError::WitnessRejected { details }
    })?;

    trace
        .steps
        .push("independent witness accepted the concrete path partition".to_string());
    trace.decision_basis.push(format!(
        "minimum path count = vertex_count - maximum_matching_size = {} - {} = {}",
        graph.vertex_count(),
        matching.size,
        graph.vertex_count() - matching.size
    ));

    tracing::info!(
        run_id = %run_id,
        minimum_path_count = cover.path_count,
        "solve accepted"
    );

    Ok(SolveResponse {
        run_id: run_id.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        graph: summary,
        matching,
        paths: cover.paths,
        path_count: cover.path_count,
        minimum_path_count: cover.path_count,
        witness,
        trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Edge;

    #[test]
    fn solves_zero_edge_dag_as_singletons() {
        let response = solve_dag_path_cover(
            "unit-zero",
            &GraphInput {
                vertices: 3,
                edges: vec![],
            },
            100,
        )
        .expect("DAG");

        assert_eq!(response.paths, vec![vec![0], vec![1], vec![2]]);
        assert_eq!(response.minimum_path_count, 3);
        assert_eq!(response.matching.size, 0);
        assert_eq!(
            response.path_count + response.matching.size,
            response.graph.vertices
        );
        assert!(response.witness.accepted);
    }

    #[test]
    fn returns_cycle_evidence_and_does_not_solve() {
        let error = solve_dag_path_cover(
            "unit-cycle",
            &GraphInput {
                vertices: 3,
                edges: vec![
                    Edge { from: 0, to: 1 },
                    Edge { from: 1, to: 2 },
                    Edge { from: 2, to: 0 },
                ],
            },
            100,
        )
        .expect_err("cycle");

        assert_eq!(error.status_code(), 422);
        match error {
            SolveError::CycleDetected { evidence, .. } => {
                assert_eq!(evidence.vertices.first(), evidence.vertices.last());
                assert_eq!(evidence.vertices, vec![0, 1, 2, 0]);
            }
            other => panic!("expected cycle evidence, got {other:?}"),
        }
    }
}
