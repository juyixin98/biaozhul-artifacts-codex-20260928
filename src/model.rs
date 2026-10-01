//! Graph request model and validated in-memory representation.

use serde::{Deserialize, Serialize};

pub type Vertex = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
}

/// User supplied graph. Vertex ids are dense integers in `0..vertices`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphInput {
    #[serde(default)]
    pub vertices: usize,
    #[serde(default)]
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphBuildError {
    VertexCountOutOfRange { actual: usize, max_vertices: usize },
    EdgeOutOfBounds { edge: Edge, vertices: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    n: usize,
    adjacency: Vec<Vec<Vertex>>,
    edge_count: usize,
    input_edge_count: usize,
}

impl Graph {
    pub fn new(input: &GraphInput, max_vertices: usize) -> Result<Self, GraphBuildError> {
        if input.vertices > max_vertices {
            return Err(GraphBuildError::VertexCountOutOfRange {
                actual: input.vertices,
                max_vertices,
            });
        }

        for edge in &input.edges {
            if edge.from >= input.vertices || edge.to >= input.vertices {
                return Err(GraphBuildError::EdgeOutOfBounds {
                    edge: *edge,
                    vertices: input.vertices,
                });
            }
        }

        let mut adjacency = vec![Vec::new(); input.vertices];
        for edge in &input.edges {
            adjacency[edge.from].push(edge.to);
        }

        for neighbors in adjacency.iter_mut() {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let edge_count: usize = adjacency.iter().map(Vec::len).sum();

        Ok(Self {
            n: input.vertices,
            adjacency,
            edge_count,
            input_edge_count: input.edges.len(),
        })
    }

    pub const fn vertex_count(&self) -> usize {
        self.n
    }

    pub const fn distinct_edge_count(&self) -> usize {
        self.edge_count
    }

    pub const fn input_edge_count(&self) -> usize {
        self.input_edge_count
    }

    pub fn neighbors(&self, vertex: Vertex) -> &[Vertex] {
        &self.adjacency[vertex]
    }

    pub fn has_edge(&self, from: Vertex, to: Vertex) -> bool {
        self.adjacency
            .get(from)
            .is_some_and(|neighbors| neighbors.binary_search(&to).is_ok())
    }

    pub fn edges(&self) -> Vec<Edge> {
        let mut edges = Vec::with_capacity(self.edge_count);
        for (from, neighbors) in self.adjacency.iter().enumerate() {
            for &to in neighbors {
                edges.push(Edge { from, to });
            }
        }
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_duplicate_edges_and_sorts_neighbors() {
        let input = GraphInput {
            vertices: 3,
            edges: vec![
                Edge { from: 0, to: 2 },
                Edge { from: 0, to: 1 },
                Edge { from: 0, to: 2 },
            ],
        };

        let graph = Graph::new(&input, 10).expect("valid graph");
        assert_eq!(graph.input_edge_count(), 3);
        assert_eq!(graph.distinct_edge_count(), 2);
        assert_eq!(graph.neighbors(0), &[1, 2]);
        assert!(graph.has_edge(0, 2));
        assert!(!graph.has_edge(2, 0));
    }

    #[test]
    fn rejects_out_of_bounds_edge() {
        let input = GraphInput {
            vertices: 2,
            edges: vec![Edge { from: 1, to: 2 }],
        };

        let err = Graph::new(&input, 10).expect_err("invalid vertex id");
        assert_eq!(
            err,
            GraphBuildError::EdgeOutOfBounds {
                edge: Edge { from: 1, to: 2 },
                vertices: 2
            }
        );
    }

    #[test]
    fn rejects_over_limit_graph() {
        let input = GraphInput {
            vertices: 4,
            edges: vec![],
        };

        assert_eq!(
            Graph::new(&input, 3),
            Err(GraphBuildError::VertexCountOutOfRange {
                actual: 4,
                max_vertices: 3
            })
        );
    }
}
