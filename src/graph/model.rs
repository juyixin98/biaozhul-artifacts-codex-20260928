use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::{EdgeId, Endpoints, Vertex};

/// Mutable compact representation used by triangle and truss kernels.
///
/// Vertex ids are dense `0..vertex_count` values. Edges are stored once in
/// canonical endpoint order.
#[derive(Debug, Clone)]
pub struct SimpleGraph {
    vertex_count: usize,
    edges: Vec<Endpoints>,
    adjacency: Vec<Vec<Vertex>>,
    edge_index: HashMap<(Vertex, Vertex), EdgeId>,
    active: Vec<bool>,
}

impl SimpleGraph {
    #[must_use]
    pub fn new(vertex_count: usize) -> Self {
        Self {
            vertex_count,
            edges: Vec::new(),
            adjacency: vec![Vec::new(); vertex_count],
            edge_index: HashMap::new(),
            active: Vec::new(),
        }
    }

    #[must_use]
    pub fn from_endpoints(
        vertex_count: usize,
        endpoints: impl IntoIterator<Item = Endpoints>,
    ) -> Self {
        let mut graph = Self::new(vertex_count);
        for edge in endpoints {
            graph.add_edge(edge.u, edge.v);
        }
        graph
    }

    pub fn add_edge(&mut self, u: Vertex, v: Vertex) -> EdgeId {
        let edge = Endpoints::new(u, v);
        if let Some(id) = self.edge_index.get(&(edge.u, edge.v)) {
            return *id;
        }
        let id = self.edges.len();
        self.edges.push(edge);
        self.active.push(true);
        self.edge_index.insert((edge.u, edge.v), id);
        self.adjacency[edge.u].push(edge.v);
        self.adjacency[edge.v].push(edge.u);
        id
    }

    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    #[must_use]
    pub fn edges(&self) -> &[Endpoints] {
        &self.edges
    }

    #[must_use]
    pub fn edge(&self, edge: EdgeId) -> Option<Endpoints> {
        self.edges.get(edge).copied()
    }

    #[must_use]
    pub fn edge_id(&self, u: Vertex, v: Vertex) -> Option<EdgeId> {
        let edge = Endpoints::new(u, v);
        self.edge_index.get(&(edge.u, edge.v)).copied()
    }

    #[must_use]
    pub fn neighbors(&self, vertex: Vertex) -> &[Vertex] {
        &self.adjacency[vertex]
    }

 #[must_use]
    pub fn is_active(&self, edge: EdgeId) -> bool {
        self.active.get(edge).copied().unwrap_or(false)
    }

    #[must_use]
    pub fn active_edge_count(&self) -> usize {
        self.active.iter().filter(|value| **value).count()
    }

    #[must_use]
    pub fn active_edges(&self) -> Vec<EdgeId> {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(edge, active)| active.then_some(edge))
            .collect()
    }

    #[must_use]
    pub fn active_neighbors(&self, vertex: Vertex) -> Vec<Vertex> {
        self.adjacency[vertex]
            .iter()
            .filter(|neighbor| {
                self.edge_id(vertex, **neighbor)
                    .is_some_and(|edge| self.is_active(edge))
            })
            .copied()
            .collect()
    }

    #[must_use]
    pub fn contains_active_edge(&self, u: Vertex, v: Vertex) -> bool {
        self.edge_id(u, v).is_some_and(|edge| self.is_active(edge))
    }

    pub fn remove_edge(&mut self, edge: EdgeId) -> bool {
        let old = self.is_active(edge);
        if old {
            self.active[edge] = false;
        }
        old
    }

    pub fn restore_edge(&mut self, edge: EdgeId) -> bool {
        let old = self.is_active(edge);
        if edge < self.active.len() {
            self.active[edge] = true;
        }
        !old && edge < self.active.len()
    }

    /// Common active neighbors of `u` and `v`.
    #[must_use]
    pub fn common_active_neighbors(&self, u: Vertex, v: Vertex) -> Vec<Vertex> {
        let left: HashSet<Vertex> = self.active_neighbors(u).into_iter().collect();
        let right: HashSet<Vertex> = self.active_neighbors(v).into_iter().collect();
        let mut result: Vec<Vertex> = left.intersection(&right).copied().collect();
        result.sort_unstable();
        result
    }

    /// Enumerates active triangles in deterministic canonical vertex order.
    #[must_use]
    pub fn triangles(&self) -> Vec<Triangle> {
        let mut triangles = Vec::new();
        for a in 0..self.vertex_count {
            for &b in self.neighbors(a) {
                if b <= a || !self.contains_active_edge(a, b) {
                    continue;
                }
                for c in self.common_active_neighbors(a, b) {
                    if c > b {
                        triangles.push(Triangle { a, b, c });
                    }
                }
            }
        }
        triangles
    }

    #[must_use]
    pub fn subgraph_snapshot(&self, edge_ids: &[EdgeId]) -> GraphSnapshot {
        let mut vertices: Vec<Vertex> = edge_ids
            .iter()
            .filter(|edge| self.is_active(**edge))
            .flat_map(|edge| {
                let endpoints = self.edges[*edge];
                [endpoints.u, endpoints.v]
            })
            .collect();
        vertices.sort_unstable();
        vertices.dedup();
        let edges = edge_ids
            .iter()
            .filter(|edge| self.is_active(**edge))
            .map(|edge| {
                let endpoints = self.edges[*edge];
                SnapshotEdge {
                    edge_id: *edge,
                    u: endpoints.u,
                    v: endpoints.v,
                }
            })
            .collect();
        GraphSnapshot { vertices, edges }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Triangle {
    pub a: Vertex,
    pub b: Vertex,
    pub c: Vertex,
}

impl Triangle {
    #[must_use]
    pub fn new(mut x: Vertex, mut y: Vertex, mut z: Vertex) -> Self {
        let mut values = [&mut x, &mut y, &mut z];
        values.sort_by_key(|value| **value);
        Self { a: x, b: y, c: z }
    }

    #[must_use]
    pub fn vertices(self) -> [Vertex; 3] {
        [self.a, self.b, self.c]
    }

    pub fn edges(self) -> impl Iterator<Item = (Vertex, Vertex)> {
        [(self.a, self.b), (self.b, self.c), (self.a, self.c)].into_iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<SnapshotEdge>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotEdge {
    pub edge_id: EdgeId,
    pub u: Vertex,
    pub v: Vertex,
}
