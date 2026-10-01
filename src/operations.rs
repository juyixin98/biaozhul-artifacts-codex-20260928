//! Wire-level batch operation types.

use serde::{Deserialize, Serialize};

use crate::model::Edge;

/// One requested graph mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Operation {
    /// Add a new edge or replace an edge with the same ID.
    UpsertEdge(Edge),
    /// Change only the weight of an existing edge.
    UpdateWeight {
        /// Existing edge ID.
        id: String,
        /// New signed weight.
        weight: i64,
    },
    /// Delete an existing edge by ID.
    DeleteEdge {
        /// Existing edge ID.
        id: String,
    },
}

impl Operation {
    /// Edge ID targeted by this operation.
    pub fn edge_id(&self) -> &str {
        match self {
            Operation::UpsertEdge(edge) => &edge.id,
            Operation::UpdateWeight { id, .. } | Operation::DeleteEdge { id } => id,
        }
    }

    /// Position of the operation inside its batch.
    pub fn index(&self) -> Option<usize> {
        None
    }
}
