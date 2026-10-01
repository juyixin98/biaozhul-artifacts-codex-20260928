use crate::error::{
    AppError, InputErrorCode, ResourceErrorCode, MAX_EDGE_COUNT, MAX_ID_LEN, MAX_NODE_COUNT,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub type Node = usize;
pub type Capacity = u64;
pub type EdgeIndex = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeSpec {
    pub from: Node,
    pub to: Node,
    pub capacity: Capacity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSpec {
    pub node_count: usize,
    pub source: Node,
    pub sink: Node,
    pub edge_ids: Vec<String>,
    pub edges: Vec<EdgeSpec>,
}

impl GraphSpec {
    pub fn validate(&self) -> Result<(), AppError> {
        if self.node_count == 0 {
            return Err(AppError::input(
                InputErrorCode::NodeCountOutOfRange,
                format!("node_count must be at least 1; got {}", self.node_count),
            ));
        }
        if self.node_count > MAX_NODE_COUNT {
            return Err(AppError::resource(
                ResourceErrorCode::GraphTooLarge,
                format!(
                    "node count {} exceeds the local resource limit {MAX_NODE_COUNT}",
                    self.node_count
                ),
            ));
        }

        if self.source >= self.node_count {
            return Err(AppError::input(
                InputErrorCode::EndpointOutOfRange,
                format!("source {} is outside [0, {})", self.source, self.node_count),
            ));
        }
        if self.sink >= self.node_count {
            return Err(AppError::input(
                InputErrorCode::EndpointOutOfRange,
                format!("sink {} is outside [0, {})", self.sink, self.node_count),
            ));
        }
        if self.source == self.sink {
            return Err(AppError::input(
                InputErrorCode::SourceEqualsSink,
                "source and sink must be different nodes",
            ));
        }
        if self.edges.len() > MAX_EDGE_COUNT {
            return Err(AppError::resource(
                ResourceErrorCode::GraphTooLarge,
                format!(
                    "edge count {} exceeds the local resource limit {MAX_EDGE_COUNT}",
                    self.edges.len()
                ),
            ));
        }
        if self.edge_ids.len() != self.edges.len() {
            return Err(AppError::input(
                InputErrorCode::DuplicateEdgeId,
                "edge_ids and edges must have the same length",
            ));
        }

        let mut ids = HashSet::with_capacity(self.edge_ids.len());
        for id in &self.edge_ids {
            if id.is_empty() {
                return Err(AppError::input(
                    InputErrorCode::EmptyEdgeId,
                    "edge ids must be non-empty",
                ));
            }
            if id.chars().count() > MAX_ID_LEN {
                return Err(AppError::input(
                    InputErrorCode::EdgeIdTooLong,
                    format!("edge id is longer than {MAX_ID_LEN} characters: {id}"),
                ));
            }
            if !ids.insert(id.clone()) {
                return Err(AppError::input(
                    InputErrorCode::DuplicateEdgeId,
                    format!("duplicate edge id: {id}"),
                ));
            }
        }

        for (edge_id, edge) in self.edge_ids.iter().zip(self.edges.iter()) {
            if edge.from >= self.node_count {
                return Err(AppError::input(
                    InputErrorCode::EndpointOutOfRange,
                    format!("edge {edge_id} has from={} outside the graph", edge.from),
                ));
            }
            if edge.to >= self.node_count {
                return Err(AppError::input(
                    InputErrorCode::EndpointOutOfRange,
                    format!("edge {edge_id} has to={} outside the graph", edge.to),
                ));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualEdge {
    pub from: Node,
    pub to: Node,
    pub remaining: Capacity,
    pub reverse: EdgeIndex,
    pub original: Option<EdgeIndex>,
}

#[derive(Debug, Clone)]
pub struct Network {
    pub spec: GraphSpec,
    pub adjacency: Vec<Vec<EdgeIndex>>,
    pub residual: Vec<ResidualEdge>,
}

impl Network {
    pub fn new(spec: GraphSpec) -> Result<Self, AppError> {
        spec.validate()?;
        let mut adjacency = vec![Vec::new(); spec.node_count];
        let mut residual = Vec::with_capacity(spec.edges.len() * 2);

        for (original_index, edge) in spec.edges.iter().enumerate() {
            let forward_index = residual.len();
            let reverse_index = forward_index + 1;

            adjacency[edge.from].push(forward_index);
            adjacency[edge.to].push(reverse_index);
            residual.push(ResidualEdge {
                from: edge.from,
                to: edge.to,
                remaining: edge.capacity,
                reverse: reverse_index,
                original: Some(original_index),
            });
            residual.push(ResidualEdge {
                from: edge.to,
                to: edge.from,
                remaining: 0,
                reverse: forward_index,
                original: None,
            });
        }

        Ok(Self {
            spec,
            adjacency,
            residual,
        })
    }

    pub fn node_count(&self) -> usize {
        self.spec.node_count
    }

    pub fn edge_flow(&self, original_edge: EdgeIndex) -> Capacity {
        let forward_index = original_edge * 2;
        let initial_capacity = self.spec.edges[original_edge].capacity;
        initial_capacity - self.residual[forward_index].remaining
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(edges: Vec<EdgeSpec>) -> GraphSpec {
        let ids = edges
            .iter()
            .enumerate()
            .map(|(i, _)| format!("e{i}"))
            .collect();
        GraphSpec {
            node_count: 2,
            source: 0,
            sink: 1,
            edge_ids: ids,
            edges,
        }
    }

    #[test]
    fn parallel_and_zero_capacity_edges_are_separate() {
        let network = Network::new(spec(vec![
            EdgeSpec {
                from: 0,
                to: 1,
                capacity: 0,
            },
            EdgeSpec {
                from: 0,
                to: 1,
                capacity: 7,
            },
        ]))
        .unwrap();

        assert_eq!(network.residual.len(), 4);
        assert_eq!(network.residual[0].reverse, 1);
        assert_eq!(network.residual[1].reverse, 0);
        assert_eq!(network.residual[2].reverse, 3);
        assert_eq!(network.residual[3].reverse, 2);
        assert_eq!(network.edge_flow(0), 0);
        assert_eq!(network.edge_flow(1), 0);
    }

    #[test]
    fn self_loops_create_matched_forward_and_residual_edges() {
        let network = Network::new(spec(vec![EdgeSpec {
            from: 0,
            to: 0,
            capacity: 5,
        }]))
        .unwrap();

        assert_eq!(network.adjacency[0], vec![0, 1]);
        assert_eq!(network.residual[0].reverse, 1);
        assert_eq!(network.residual[1].reverse, 0);
    }
}
