use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::algorithm::DecompositionReport;
use crate::graph::{Endpoints, GraphSnapshot, Triangle};
use crate::witness::Witness;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum WitnessCheckError {
    MissingWitness { edge_id: usize, witness_id: usize },
    DuplicateWitnessId { witness_id: usize },
    EdgeOutsideWitness { edge_id: usize, witness_id: usize },
    ThresholdMismatch {
        witness_id: usize,
        truss_level: usize,
    },
    UnsupportedEdge {
        witness_id: usize,
        edge_id: usize,
        support: usize,
        required: usize,
    },
}

impl WitnessCheckError {
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::MissingWitness { .. } => "missing_witness",
            Self::DuplicateWitnessId { .. } => "duplicate_witness_id",
            Self::EdgeOutsideWitness { .. } => "edge_outside_witness",
            Self::ThresholdMismatch { .. } => "threshold_mismatch",
            Self::UnsupportedEdge { .. } => "unsupported_edge",
        }
    }
}

/// Checks the positive witnesses independently from the peeling state machine.
///
/// A witness for level k must contain the claimed edge, and every edge in that
/// witness must have at least k-2 triangles wholly inside the witness.
#[must_use]
pub fn check_report_witnesses(report: &DecompositionReport) -> Result<(), WitnessCheckError> {
    let mut by_id = HashMap::new();
    for witness in &report.witnesses {
        if by_id.insert(witness.witness_id, witness).is_some() {
            return Err(WitnessCheckError::DuplicateWitnessId {
                witness_id: witness.witness_id,
            });
        }
    }

    for edge in &report.levels {
        let witness = by_id.get(&edge.witness_id).ok_or(
            WitnessCheckError::MissingWitness {
                edge_id: edge.edge_id,
                witness_id: edge.witness_id,
            },
        )?;
        if witness.support_threshold != witness.truss_level.saturating_sub(2) {
            return Err(WitnessCheckError::ThresholdMismatch {
                witness_id: witness.witness_id,
                truss_level: witness.truss_level,
            });
        }
        if !witness
            .subgraph
            .edges
            .iter()
            .any(|snapshot| snapshot.edge_id == edge.edge_id)
        {
            return Err(WitnessCheckError::EdgeOutsideWitness {
                edge_id: edge.edge_id,
                witness_id: witness.witness_id,
            });
        }
        check_support_threshold(witness)?;
    }

    Ok(())
}

fn check_support_threshold(witness: &Witness) -> Result<(), WitnessCheckError> {
    let triangles = enumerate_snapshot_triangles(&witness.subgraph);
    let mut support: HashMap<usize, usize> = witness
        .subgraph
        .edges
        .iter()
        .map(|edge| (edge.edge_id, 0))
        .collect();

    for triangle in triangles {
        for (u, v) in triangle.edges() {
            let edge_id = witness
                .subgraph
                .edges
                .iter()
                .find(|edge| {
                    let endpoints = Endpoints::new(u, v);
                    edge.u == endpoints.u && edge.v == endpoints.v
                })
                .map(|edge| edge.edge_id);
            if let Some(edge_id) = edge_id {
                *support.entry(edge_id).or_insert(0) += 1;
            }
        }
    }

    for edge in &witness.subgraph.edges {
        let support = support[&edge.edge_id];
        if support < witness.support_threshold {
            return Err(WitnessCheckError::UnsupportedEdge {
                witness_id: witness.witness_id,
                edge_id: edge.edge_id,
                support,
                required: witness.support_threshold,
            });
        }
    }
    Ok(())
}

#[must_use]
fn enumerate_snapshot_triangles(snapshot: &GraphSnapshot) -> Vec<Triangle> {
    let edge_set: HashSet<(usize, usize)> = snapshot
        .edges
        .iter()
        .map(|edge| (edge.u, edge.v))
        .collect();
    let mut vertices = snapshot.vertices.clone();
    vertices.sort_unstable();
    vertices.dedup();

    let mut triangles = Vec::new();
    for (index, &a) in vertices.iter().enumerate() {
        for &b in vertices.iter().skip(index + 1) {
            if !edge_set.contains(&(a, b)) {
                continue;
            }
            for &c in vertices.iter().skip(index + 2) {
                if c > b && edge_set.contains(&(a, c)) && edge_set.contains(&(b, c)) {
                    triangles.push(Triangle { a, b, c });
                }
            }
        }
    }
    triangles
}
