use crate::MAX_EDGES;
use crate::MAX_NODES;
use serde::{Deserialize, Serialize};

use super::errors::{InputError, ResourceLimit};

pub type Capacity = u64;
pub type NodeId = usize;

/// A directed edge supplied by the caller.
///
/// Multiple edges with the same `(from, to)` are parallel original edges and
/// remain distinct in output. A zero capacity is valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeSpec {
    pub from: NodeId,
    pub to: NodeId,
    pub capacity: Capacity,
}

impl EdgeSpec {
    #[must_use]
    pub fn new(from: NodeId, to: NodeId, capacity: Capacity) -> Self {
        Self { from, to, capacity }
    }
}

/// External graph request contract. Node ids are zero based.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkSpec {
    pub node_count: usize,
    pub source: NodeId,
    pub sink: NodeId,
    #[serde(default)]
    pub edges: Vec<EdgeSpec>,
}

/// Validated immutable network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Network {
    node_count: usize,
    source: NodeId,
    sink: NodeId,
    edges: Vec<EdgeSpec>,
}

impl Network {
    pub fn build(spec: NetworkSpec) -> Result<Self, InputError> {
        if spec.node_count == 0 {
            return Err(InputError::EmptyNodeSet);
        }
        if spec.node_count > MAX_NODES {
            return Err(InputError::LimitExceeded(ResourceLimit {
                name: "nodes".to_string(),
                limit: MAX_NODES as u64,
                actual: spec.node_count as u64,
            }));
        }
        if spec.edges.len() > MAX_EDGES {
            return Err(InputError::LimitExceeded(ResourceLimit {
                name: "edges".to_string(),
                limit: MAX_EDGES as u64,
                actual: spec.edges.len() as u64,
            }));
        }
        if spec.source >= spec.node_count {
            return Err(InputError::SourceOutOfRange {
                source: spec.source,
                node_count: spec.node_count,
            });
        }
        if spec.sink >= spec.node_count {
            return Err(InputError::SinkOutOfRange {
                sink: spec.sink,
                node_count: spec.node_count,
            });
        }
        if spec.source == spec.sink {
            return Err(InputError::SourceEqualsSink { node: spec.source });
        }
        for (index, edge) in spec.edges.iter().enumerate() {
            if edge.from >= spec.node_count {
                return Err(InputError::EdgeEndpointOutOfRange {
                    index,
                    endpoint: "from".to_string(),
                    node: edge.from,
                    node_count: spec.node_count,
                });
            }
            if edge.to >= spec.node_count {
                return Err(InputError::EdgeEndpointOutOfRange {
                    index,
                    endpoint: "to".to_string(),
                    node: edge.to,
                    node_count: spec.node_count,
                });
            }
        }

        Ok(Self {
            node_count: spec.node_count,
            source: spec.source,
            sink: spec.sink,
            edges: spec.edges,
        })
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    #[must_use]
    pub fn source(&self) -> NodeId {
        self.source
    }

    #[must_use]
    pub fn sink(&self) -> NodeId {
        self.sink
    }

    #[must_use]
    pub fn edges(&self) -> &[EdgeSpec] {
        &self.edges
    }

    #[must_use]
    pub fn spec(&self) -> NetworkSpec {
        NetworkSpec {
            node_count: self.node_count,
            source: self.source,
            sink: self.sink,
            edges: self.edges.clone(),
        }
    }
}
