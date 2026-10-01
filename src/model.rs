//! Immutable value types and the mutable undirected graph container.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Vertex identifier. String IDs make request logs stable and easy to trace.
pub type Vertex = String;
/// Edge identifier. Equal-weight ties are resolved by this identifier.
pub type EdgeId = String;
/// Edge weight. Signed integers intentionally support negative edge weights.
pub type Weight = i64;

/// One undirected weighted edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    /// Stable edge identifier.
    pub id: EdgeId,
    /// First endpoint. Edges are undirected, so endpoint order has no meaning.
    pub u: Vertex,
    /// Second endpoint.
    pub v: Vertex,
    /// Signed edge weight.
    pub weight: Weight,
}

impl Edge {
    /// Returns true when the edge connects a vertex to itself.
    pub fn is_self_loop(&self) -> bool {
        self.u == self.v
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Edge belongs to the maintained spanning forest.
    Tree,
    /// Edge is present but is not a forest edge.
    NonTree,
    /// Self-loop edges are retained in graph state but excluded from forests.
    IgnoredSelfLoop,
}

/// A complete, versioned graph snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Graph {
    version: u64,
    edges: BTreeMap<EdgeId, Edge>,
    vertices: BTreeSet<Vertex>,
    adjacency: BTreeMap<Vertex, BTreeSet<(EdgeId, Vertex)>>,
}

impl Graph {
    /// Create an empty graph with version zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current complete-graph version.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Set the version. The service owns version transitions; tests may seed it.
    pub fn set_version(&mut self, version: u64) {
        self.version = version;
    }

    /// Look up an edge by ID.
    pub fn edge(&self, id: &str) -> Option<&Edge> {
        self.edges.get(id)
    }

    /// Iterate over all current edges in stable ID order.
    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.values()
    }

    /// Iterate over all vertices, including vertices made isolated by deletion.
    pub fn vertices(&self) -> impl Iterator<Item = &Vertex> {
        self.vertices.iter()
    }

    /// Iterate over non-self-loop incident edges of `vertex`.
    pub fn incident(&self, vertex: &str) -> impl Iterator<Item = &(EdgeId, Vertex)> {
        self.adjacency
            .get(vertex)
            .into_iter()
            .flat_map(|neighbors| neighbors.iter())
    }

    /// Add or replace an edge. Replacing also moves its adjacency endpoints.
    pub fn upsert_edge(&mut self, edge: Edge) {
        if let Some(old) = self.edges.insert(edge.id.clone(), edge.clone()) {
            self.remove_adjacency(&old);
        }

        self.vertices.insert(edge.u.clone());
        self.vertices.insert(edge.v.clone());
        if !edge.is_self_loop() {
            self.adjacency
                .entry(edge.u.clone())
                .or_default()
                .insert((edge.id.clone(), edge.v.clone()));
            self.adjacency
                .entry(edge.v.clone())
                .or_default()
                .insert((edge.id.clone(), edge.u.clone()));
        }
    }

    /// Remove an edge. Isolated vertices remain explicit vertices.
    pub fn remove_edge(&mut self, id: &str) -> Option<Edge> {
        let edge = self.edges.remove(id)?;
        self.remove_adjacency(&edge);
        Some(edge)
    }

    /// Replace an existing edge in place, returning the previous value.
    pub fn replace_edge(&mut self, edge: Edge) -> Option<Edge> {
        if !self.edges.contains_key(&edge.id) {
            return None;
        }
        let old = self.edges.remove(&edge.id)?;
        self.remove_adjacency(&old);
        self.upsert_edge(edge.clone());
        Some(old)
    }

    /// Classify an edge relative to a set of forest edge IDs.
    pub fn classify(&self, id: &str, forest_ids: &BTreeSet<EdgeId>) -> Option<EdgeKind> {
        let edge = self.edges.get(id)?;
        if edge.is_self_loop() {
            Some(EdgeKind::IgnoredSelfLoop)
        } else if forest_ids.contains(id) {
            Some(EdgeKind::Tree)
        } else {
            Some(EdgeKind::NonTree)
        }
    }

    fn remove_adjacency(&mut self, edge: &Edge) {
        if let Some(neighbors) = self.adjacency.get_mut(&edge.u) {
            neighbors.remove(&(edge.id.clone(), edge.v.clone()));
        }
        if edge.u != edge.v {
            if let Some(neighbors) = self.adjacency.get_mut(&edge.v) {
                neighbors.remove(&(edge.id.clone(), edge.u.clone()));
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn edge_count(&self) -> usize {
        self.edges.len()
    }

    #[cfg(test)]
    pub(crate) fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(id: &str, u: &str, v: &str, weight: Weight) -> Edge {
        Edge {
            id: id.to_string(),
            u: u.to_string(),
            v: v.to_string(),
            weight,
        }
    }

    #[test]
    fn keeps_loops_but_has_no_loop_adjacency() {
        let mut graph = Graph::new();
        graph.upsert_edge(edge("l", "a", "a", 1));
        graph.upsert_edge(edge("e", "a", "b", 2));

        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.incident("a").count(), 1);
        assert_eq!(
            graph.classify("l", &BTreeSet::new()),
            Some(EdgeKind::IgnoredSelfLoop)
        );
    }

    #[test]
    fn replacing_edge_moves_adjacency() {
        let mut graph = Graph::new();
        graph.upsert_edge(edge("e", "a", "b", 1));
        graph.replace_edge(edge("e", "a", "c", 3));

        assert_eq!(graph.edge("e").map(|edge| edge.v.as_str()), Some("c"));
        assert_eq!(graph.incident("b").count(), 0);
        assert_eq!(graph.edge("e").map(|edge| edge.u.as_str()), Some("a"));
        assert_eq!(graph.vertex_count(), 3);
    }
}
