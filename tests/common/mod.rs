#![allow(dead_code)]

use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use truss_backend::graph::GraphInput;

pub fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

pub fn load_fixture(name: &str) -> GraphInput {
    let data = fs::read_to_string(fixture_path(name)).unwrap();
    serde_json::from_str(&data).unwrap()
}

pub fn load_fixture_json(name: &str) -> Value {
    let data = fs::read_to_string(fixture_path(name)).unwrap();
    serde_json::from_str(&data).unwrap()
}

/// Independent deliberately slow reference model.
///
/// It normalizes edges with a set, enumerates every vertex triple for triangle
/// counts, and recomputes support from the active set throughout truss peeling.
/// It intentionally does not call the production kernel.
#[derive(Debug, Default)]
pub struct ReferenceGraph {
    node_count: usize,
    edges: Vec<(usize, usize)>,
    edge_ids: Vec<String>,
}

impl ReferenceGraph {
    pub fn from_input(input: &GraphInput) -> Self {
        let mut id_to_index = std::collections::HashMap::new();
        for (index, node) in input.nodes.iter().enumerate() {
            id_to_index.insert(node.id.clone(), index);
        }

        let mut seen_pairs = HashSet::new();
        let mut edges = Vec::new();
        let mut edge_ids = Vec::new();

        for (index, edge) in input.edges.iter().enumerate() {
            assert_ne!(edge.u, edge.v, "reference does not accept self loops");
            let mut pair = (id_to_index[&edge.u], id_to_index[&edge.v]);
            if pair.0 > pair.1 {
                std::mem::swap(&mut pair.0, &mut pair.1);
            }
            if seen_pairs.insert(pair) {
                edges.push(pair);
                edge_ids.push(edge.id.clone().unwrap_or_else(|| format!("e{index}")));
            }
        }

        Self {
            node_count: input.nodes.len(),
            edges,
            edge_ids,
        }
    }

    pub fn triangle_count(&self) -> usize {
        let active: HashSet<(usize, usize)> = self.edges.iter().copied().collect();
        let mut count = 0;
        for a in 0..self.node_count {
            for b in (a + 1)..self.node_count {
                for c in (b + 1)..self.node_count {
                    if active.contains(&(a, b))
                        && active.contains(&(a, c))
                        && active.contains(&(b, c))
                    {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    pub fn edge_support(&self, edge_index: usize) -> usize {
        let active: HashSet<(usize, usize)> = self.edges.iter().copied().collect();
        self.support_in(edge_index, &active)
    }

    pub fn truss_levels(&self) -> std::collections::HashMap<String, usize> {
        let mut unassigned: HashSet<(usize, usize)> = self.edges.iter().copied().collect();
        let mut levels = std::collections::HashMap::new();

        let mut active: HashSet<(usize, usize)> = unassigned.clone();

        for k in 3..=self.node_count {
            loop {
                let mut doomed = Vec::new();
                for edge in &unassigned {
                    if active.contains(edge) && self.support_in_pair(*edge, &active) < k - 2 {
                        doomed.push(*edge);
                    }
                }
                if doomed.is_empty() {
                    break;
                }
                for edge in doomed {
                    active.remove(&edge);
                }
            }

            let survived: HashSet<(usize, usize)> =
                unassigned.intersection(&active).copied().collect();
            for edge in &survived {
                unassigned.remove(edge);
                levels.insert(*edge, k);
            }
        }

        for edge in &unassigned {
            levels.insert(*edge, 2);
        }

        self.edges
            .iter()
            .zip(&self.edge_ids)
            .filter_map(|(edge, id)| levels.get(edge).map(|level| (id.clone(), *level)))
            .collect()
    }

    fn support_in(&self, edge_index: usize, active: &HashSet<(usize, usize)>) -> usize {
        self.support_in_pair(self.edges[edge_index], active)
    }

    fn support_in_pair(&self, (u, v): (usize, usize), active: &HashSet<(usize, usize)>) -> usize {
        (0..self.node_count)
            .filter(|w| {
                *w != u
                    && *w != v
                    && active.contains(&ordered(u, *w))
                    && active.contains(&ordered(v, *w))
            })
            .count()
    }
}

fn ordered(mut a: usize, mut b: usize) -> (usize, usize) {
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    (a, b)
}
