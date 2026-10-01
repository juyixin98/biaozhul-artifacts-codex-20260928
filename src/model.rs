use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub u: i64,
    pub v: i64,
    pub weight: i64,
}

impl Edge {
    pub fn new(id: impl Into<String>, u: i64, v: i64, weight: i64) -> Self {
        // Store undirected endpoints in one canonical order.
        let (u, v) = if u <= v { (u, v) } else { (v, u) };
        Self {
            id: id.into(),
            u,
            v,
            weight,
        }
    }

    pub fn is_self_loop(&self) -> bool {
        self.u == self.v
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidEdgeReason {
    EmptyId,
    IdTooLong,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    DuplicateEdgeId(String),
    EdgeNotFound(String),
    InvalidEdge {
        edge_id: String,
        reason: InvalidEdgeReason,
    },
    EdgeLimitExceeded {
        limit: usize,
    },
    ArithmeticOverflow,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateEdgeId(id) => write!(f, "edge id `{id}` already exists"),
            Self::EdgeNotFound(id) => write!(f, "edge id `{id}` does not exist"),
            Self::InvalidEdge {
                edge_id,
                reason: InvalidEdgeReason::EmptyId,
            } => {
                write!(f, "edge id `{edge_id}` is empty or contains whitespace")
            }
            Self::InvalidEdge {
                edge_id,
                reason: InvalidEdgeReason::IdTooLong,
            } => {
                write!(f, "edge id `{edge_id}` is too long")
            }
            Self::EdgeLimitExceeded { limit } => {
                write!(f, "graph edge limit of {limit} was exceeded")
            }
            Self::ArithmeticOverflow => write!(f, "total weight arithmetic overflowed"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Graph {
    pub vertices: BTreeSet<i64>,
    pub edges: BTreeMap<String, Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_edge(
        &mut self,
        edge: Edge,
        max_edges: usize,
        max_edge_id_length: usize,
    ) -> Result<(), ModelError> {
        if edge.id.is_empty() || edge.id.chars().any(char::is_whitespace) {
            return Err(ModelError::InvalidEdge {
                edge_id: edge.id.clone(),
                reason: InvalidEdgeReason::EmptyId,
            });
        }
        if edge.id.len() > max_edge_id_length {
            return Err(ModelError::InvalidEdge {
                edge_id: edge.id.clone(),
                reason: InvalidEdgeReason::IdTooLong,
            });
        }
        if self.edges.contains_key(&edge.id) {
            return Err(ModelError::DuplicateEdgeId(edge.id));
        }
        if self.edges.len() >= max_edges {
            return Err(ModelError::EdgeLimitExceeded { limit: max_edges });
        }
        self.vertices.insert(edge.u);
        self.vertices.insert(edge.v);
        self.edges.insert(edge.id.clone(), edge);
        Ok(())
    }

    pub fn remove_edge(&mut self, id: &str) -> Result<Edge, ModelError> {
        self.edges
            .remove(id)
            .ok_or_else(|| ModelError::EdgeNotFound(id.to_string()))
    }

    pub fn set_weight(&mut self, id: &str, weight: i64) -> Result<Edge, ModelError> {
        let old = self
            .edges
            .get(id)
            .ok_or_else(|| ModelError::EdgeNotFound(id.to_string()))?
            .clone();
        self.edges.get_mut(id).expect("checked above").weight = weight;
        Ok(old)
    }
}
