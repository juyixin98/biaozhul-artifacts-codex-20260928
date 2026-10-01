use serde::{Deserialize, Serialize};

use crate::algorithm::WitnessAssignment;
use crate::graph::{EdgeId, GraphSnapshot, SimpleGraph, Vertex};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Witness {
    pub witness_id: usize,
    pub truss_level: usize,
    pub support_threshold: usize,
    pub subgraph: GraphSnapshot,
}

#[derive(Debug)]
struct Dsu {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl Dsu {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, mut vertex: Vertex) -> Vertex {
        while self.parent[vertex] != vertex {
            self.parent[vertex] = self.parent[self.parent[vertex]];
            vertex = self.parent[vertex];
        }
        vertex
    }

    fn union(&mut self, left: Vertex, right: Vertex) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        match self.rank[left_root].cmp(&self.rank[right_root]) {
            std::cmp::Ordering::Less => self.parent[left_root] = right_root,
            std::cmp::Ordering::Greater => self.parent[right_root] = left_root,
            std::cmp::Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] += 1;
            }
        }
    }
}

/// Build one witness per connected component of the graph active before a
/// peeling stage.
///
/// Every edge removed at `stage_level` belongs to one of these components. The
/// component is a valid `stage_level`-truss at that moment because all active
/// edge supports are at least `stage_level - 2` until the first removal of the
/// stage. Edge deletions only begin after these snapshots are taken lazily.
#[must_use]
pub fn assign_stage_witnesses(
    graph: &SimpleGraph,
    _stage_level: usize,
    next_witness_id: usize,
) -> Vec<WitnessAssignment> {
    let active_edges = graph.active_edges();
    if active_edges.is_empty() {
        return Vec::new();
    }

    let mut dsu = Dsu::new(graph.vertex_count());
    for edge in &active_edges {
        let endpoints = graph.edge(*edge).expect("active edge exists");
        dsu.union(endpoints.u, endpoints.v);
    }

    let mut root_order: Vec<Vertex> = Vec::new();
    let mut root_to_witness: std::collections::HashMap<Vertex, usize> =
        std::collections::HashMap::new();
    let mut edges_by_component: std::collections::HashMap<Vertex, Vec<EdgeId>> =
        std::collections::HashMap::new();

    for edge in active_edges {
        let endpoints = graph.edge(edge).expect("active edge exists");
        let root = dsu.find(endpoints.u);
        if !root_to_witness.contains_key(&root) {
            root_to_witness.insert(root, next_witness_id + root_order.len());
            root_order.push(root);
        }
        edges_by_component.entry(root).or_default().push(edge);
    }

    root_order
        .into_iter()
        .map(|root| {
            let mut component_edges = edges_by_component.remove(&root).unwrap_or_default();
            component_edges.sort_unstable();
            WitnessAssignment {
                witness_id: root_to_witness[&root],
                subgraph: graph.subgraph_snapshot(&component_edges),
            }
        })
        .collect()
}
