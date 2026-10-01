#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefEdge {
    pub id: String,
    pub u: i64,
    pub v: i64,
    pub weight: i64,
}

#[derive(Debug, Clone, Default)]
pub struct RefGraph {
    pub edges: BTreeMap<String, RefEdge>,
}

impl RefGraph {
    pub fn add_edge(&mut self, id: &str, u: i64, v: i64, weight: i64) {
        let (u, v) = if u <= v { (u, v) } else { (v, u) };
        self.edges.insert(
            id.to_string(),
            RefEdge {
                id: id.to_string(),
                u,
                v,
                weight,
            },
        );
    }

    pub fn delete_edge(&mut self, id: &str) {
        self.edges.remove(id);
    }

    pub fn set_weight(&mut self, id: &str, weight: i64) {
        if let Some(edge) = self.edges.get_mut(id) {
            edge.weight = weight;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefMsf {
    pub forest: BTreeSet<String>,
    pub total_weight: i128,
}

pub fn kruskal(graph: &RefGraph) -> RefMsf {
    let mut vertices: BTreeSet<i64> = BTreeSet::new();
    for edge in graph.edges.values() {
        vertices.insert(edge.u);
        vertices.insert(edge.v);
    }

    let index: BTreeMap<_, _> = vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (*vertex, index))
        .collect();
    let mut parent: Vec<usize> = (0..vertices.len()).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] == x {
            x
        } else {
            parent[x] = find(parent, parent[x]);
            parent[x]
        }
    }

    fn union(parent: &mut [usize], a: usize, b: usize) -> bool {
        let ar = find(parent, a);
        let br = find(parent, b);
        if ar == br {
            return false;
        }
        parent[br] = ar;
        true
    }

    let mut ordered: Vec<_> = graph.edges.values().collect();
    ordered.sort_by(|a, b| (a.weight, a.id.as_str()).cmp(&(b.weight, b.id.as_str())));

    let mut forest = BTreeSet::new();
    let mut total_weight = 0i128;
    for edge in ordered {
        if edge.u == edge.v {
            continue;
        }
        let ai = *index.get(&edge.u).expect("vertex indexed");
        let bi = *index.get(&edge.v).expect("vertex indexed");
        if union(&mut parent, ai, bi) {
            forest.insert(edge.id.clone());
            total_weight += edge.weight as i128;
        }
    }

    RefMsf {
        forest,
        total_weight,
    }
}

pub fn assert_acyclic(graph: &RefGraph, forest: &BTreeSet<String>) {
    let mut vertices: BTreeSet<i64> = BTreeSet::new();
    let mut forest_edges: Vec<&RefEdge> = Vec::new();
    for id in forest {
        let edge = graph.edges.get(id).expect("forest edge exists");
        assert_ne!(edge.u, edge.v, "self-loop {id} cannot belong to a forest");
        vertices.insert(edge.u);
        vertices.insert(edge.v);
        forest_edges.push(edge);
    }
    assert!(
        forest_edges.len()
            <= vertices
                .len()
                .saturating_sub(if vertices.is_empty() { 0 } else { 1 }),
        "forest has too many edges"
    );

    let index: BTreeMap<_, _> = vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (*vertex, index))
        .collect();
    let mut parent: Vec<usize> = (0..vertices.len()).collect();
    for edge in forest_edges {
        let ai = *index.get(&edge.u).expect("vertex indexed");
        let bi = *index.get(&edge.v).expect("vertex indexed");
        let ar = {
            let mut x = ai;
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        };
        let br = {
            let mut x = bi;
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        };
        assert_ne!(ar, br, "forest edge {} closes a cycle", edge.id);
        parent[br] = ar;
    }
}

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        // xorshift64: deterministic and sufficient for synthetic fixtures.
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
}
