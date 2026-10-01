#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

pub type Edge = (usize, usize);

#[derive(Debug, Clone)]
pub struct ReferenceGraph {
    vertex_count: usize,
    edges: BTreeSet<Edge>,
}

impl ReferenceGraph {
    pub fn new(vertex_count: usize) -> Self {
        Self {
            vertex_count,
            edges: BTreeSet::new(),
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        assert_ne!(u, v, "reference oracle does not accept self loops");
        let (u, v) = if u < v { (u, v) } else { (v, u) };
        self.edges.insert((u, v));
    }

    pub fn edges(&self) -> Vec<Edge> {
        self.edges.iter().copied().collect()
    }

    pub fn triangles(&self) -> BTreeSet<(usize, usize, usize)> {
        let mut triangles = BTreeSet::new();
        for a in 0..self.vertex_count {
            for b in (a + 1)..self.vertex_count {
                if !self.edges.contains(&(a, b)) {
                    continue;
                }
                for c in (b + 1)..self.vertex_count {
                    if self.edges.contains(&(a, c)) && self.edges.contains(&(b, c)) {
                        triangles.insert((a, b, c));
                    }
                }
            }
        }
        triangles
    }

    /// Recompute support from scratch for every currently present edge.
    pub fn support(&self) -> BTreeMap<Edge, usize> {
        let triangles = self.triangles();
        let mut support: BTreeMap<Edge, usize> =
            self.edges.iter().map(|edge| (*edge, 0)).collect();
        for (a, b, c) in triangles {
            for edge in [(a, b), (b, c), (a, c)] {
                *support.entry(edge).or_insert(0) += 1;
            }
        }
        support
    }

    /// Reference k-truss assignment by repeated full recomputation.
    ///
    /// At each level k, remove every edge whose support is below k-2. Support
    /// is recomputed globally after each batch. The production kernel is never
    /// called by this code.
    pub fn truss_levels(mut self) -> BTreeMap<Edge, usize> {
        let mut levels = BTreeMap::new();
        let mut level: usize = 2;

        while !self.edges.is_empty() {
            loop {
                let support = self.support();
                let removed: Vec<Edge> = support
                    .iter()
                    .filter_map(|(edge, edge_support)| {
                        (*edge_support < level.saturating_sub(2)).then_some(*edge)
                    })
                    .collect();
                if removed.is_empty() {
                    break;
                }
                for edge in removed {
                    self.edges.remove(&edge);
                }
            }

            loop {
                let support = self.support();
                let removed: Vec<Edge> = support
                    .iter()
                    .filter_map(|(edge, edge_support)| {
                        (*edge_support <= level.saturating_sub(2)).then_some(*edge)
                    })
                    .collect();
                if removed.is_empty() {
                    break;
                }
                for edge in removed {
                    self.edges.remove(&edge);
                    levels.insert(edge, level);
                }
            }
            level += 1;
        }

        levels
    }
}

pub fn all_graphs(vertex_count: usize) -> Vec<ReferenceGraph> {
    let mut pairs = Vec::new();
    for u in 0..vertex_count {
        for v in (u + 1)..vertex_count {
            pairs.push((u, v));
        }
    }
    let mut graphs = vec![ReferenceGraph::new(vertex_count)];
    for edge in pairs {
        let mut next = Vec::new();
        for graph in &graphs {
            next.push(graph.clone());
            let mut with_edge = graph.clone();
            with_edge.add_edge(edge.0, edge.1);
            next.push(with_edge);
        }
        graphs = next;
    }
    graphs
}
