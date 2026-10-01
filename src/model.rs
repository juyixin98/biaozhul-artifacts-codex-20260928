use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Graph {
    pub vertex_count: usize,
    pub edges: Vec<(usize, usize)>,
    pub adjacency: Vec<Vec<usize>>,
    pub indegree: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphBuildError {
    VertexLimitExceeded { limit: usize, actual: usize },
    EdgeLimitExceeded { limit: usize, actual: usize },
    SelfLoop { vertex: usize },
    DuplicateEdge { from: usize, to: usize },
}

impl Graph {
    pub fn new(
        vertex_count: usize,
        raw_edges: Vec<(usize, usize)>,
        max_edges: usize,
    ) -> Result<Self, GraphBuildError> {
        if vertex_count > 1000 {
            return Err(GraphBuildError::VertexLimitExceeded {
                limit: 1000,
                actual: vertex_count,
            });
        }
        if raw_edges.len() > max_edges {
            return Err(GraphBuildError::EdgeLimitExceeded {
                limit: max_edges,
                actual: raw_edges.len(),
            });
        }

        let mut adjacency = vec![Vec::new(); vertex_count];
        let mut indegree = vec![0usize; vertex_count];
        let mut seen = HashSet::with_capacity(raw_edges.len());

        for &(from, to) in &raw_edges {
            if from >= vertex_count || to >= vertex_count {
                panic!("edge endpoint outside graph: {from} -> {to}");
            }
            if from == to {
                return Err(GraphBuildError::SelfLoop { vertex: from });
            }
            if !seen.insert((from, to)) {
                return Err(GraphBuildError::DuplicateEdge { from, to });
            }
            adjacency[from].push(to);
            indegree[to] += 1;
        }

        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
        }

        Ok(Self {
            vertex_count,
            edges: raw_edges,
            adjacency,
            indegree,
        })
    }

    pub fn sources(&self) -> Vec<usize> {
        (0..self.vertex_count)
            .filter(|v| self.indegree[*v] == 0)
            .collect()
    }

    pub fn sinks(&self) -> Vec<usize> {
        (0..self.vertex_count)
            .filter(|v| self.adjacency[*v].is_empty())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledGraph {
    pub vertices: Vec<String>,
    pub edges: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LabeledGraphError {
    EmptyVertexId {
        index: usize,
    },
    EmptyEdgeEndpoint {
        edge_index: usize,
        side: EdgeSide,
    },
    DuplicateVertex {
        vertex: String,
        first_index: usize,
        duplicate_index: usize,
    },
    UnknownEdgeEndpoint {
        edge_index: usize,
        vertex: String,
        side: EdgeSide,
    },
    SelfLoop {
        edge_index: usize,
        vertex: String,
    },
    DuplicateEdge {
        edge_index: usize,
        from: String,
        to: String,
    },
    VertexLimitExceeded {
        limit: usize,
        actual: usize,
    },
    EdgeLimitExceeded {
        limit: usize,
        actual: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeSide {
    From,
    To,
}

impl EdgeSide {
    pub fn as_str(self) -> &'static str {
        match self {
            EdgeSide::From => "from",
            EdgeSide::To => "to",
        }
    }
}

impl LabeledGraph {
    pub fn into_core_graph(
        self,
        max_vertices: usize,
        max_edges: usize,
    ) -> Result<(Graph, Vec<String>), LabeledGraphError> {
        if self.vertices.len() > max_vertices {
            return Err(LabeledGraphError::VertexLimitExceeded {
                limit: max_vertices,
                actual: self.vertices.len(),
            });
        }
        if self.edges.len() > max_edges {
            return Err(LabeledGraphError::EdgeLimitExceeded {
                limit: max_edges,
                actual: self.edges.len(),
            });
        }

        let mut ids = HashMap::with_capacity(self.vertices.len());
        let mut labels = Vec::with_capacity(self.vertices.len());
        for (index, vertex) in self.vertices.into_iter().enumerate() {
            if vertex.trim().is_empty() {
                return Err(LabeledGraphError::EmptyVertexId { index });
            }
            if let Some(first_index) = ids.insert(vertex.clone(), index) {
                return Err(LabeledGraphError::DuplicateVertex {
                    vertex,
                    first_index,
                    duplicate_index: index,
                });
            }
            labels.push(vertex);
        }

        let mut numeric_edges = Vec::with_capacity(self.edges.len());
        let mut duplicate_guard: HashSet<(usize, usize)> = HashSet::with_capacity(self.edges.len());

        for (edge_index, (from_label, to_label)) in self.edges.into_iter().enumerate() {
            if from_label.trim().is_empty() {
                return Err(LabeledGraphError::EmptyEdgeEndpoint {
                    edge_index,
                    side: EdgeSide::From,
                });
            }
            if to_label.trim().is_empty() {
                return Err(LabeledGraphError::EmptyEdgeEndpoint {
                    edge_index,
                    side: EdgeSide::To,
                });
            }

            let from =
                *ids.get(&from_label)
                    .ok_or_else(|| LabeledGraphError::UnknownEdgeEndpoint {
                        edge_index,
                        vertex: from_label.clone(),
                        side: EdgeSide::From,
                    })?;
            let to = *ids
                .get(&to_label)
                .ok_or_else(|| LabeledGraphError::UnknownEdgeEndpoint {
                    edge_index,
                    vertex: to_label.clone(),
                    side: EdgeSide::To,
                })?;

            if from == to {
                return Err(LabeledGraphError::SelfLoop {
                    edge_index,
                    vertex: from_label,
                });
            }
            if !duplicate_guard.insert((from, to)) {
                return Err(LabeledGraphError::DuplicateEdge {
                    edge_index,
                    from: from_label,
                    to: to_label,
                });
            }
            numeric_edges.push((from, to));
        }

        Graph::new(labels.len(), numeric_edges, max_edges)
            .map(|graph| (graph, labels))
            .map_err(|error| match error {
                GraphBuildError::VertexLimitExceeded { limit, actual } => {
                    LabeledGraphError::VertexLimitExceeded { limit, actual }
                }
                GraphBuildError::EdgeLimitExceeded { limit, actual } => {
                    LabeledGraphError::EdgeLimitExceeded { limit, actual }
                }
                GraphBuildError::SelfLoop { .. } | GraphBuildError::DuplicateEdge { .. } => {
                    unreachable!(
                        "label validation rejects loops and duplicate edges before Graph::new"
                    )
                }
            })
    }

    pub fn label_set(vertices: &[String]) -> BTreeSet<String> {
        vertices.iter().cloned().collect()
    }
}
