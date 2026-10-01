use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::errors::{InputError, LIMIT_RAW_EDGES, LIMIT_VERTICES};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeInput {
    pub u: String,
    pub v: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphInput {
    #[serde(default)]
    pub vertices: Vec<String>,
    #[serde(default)]
    pub edges: Vec<EdgeInput>,
}

#[derive(Debug, Clone)]
pub struct Graph {
    names: Vec<String>,
    index: BTreeMap<String, usize>,
    adjacency: Vec<BTreeSet<usize>>,
    edge_count: usize,
    duplicate_edge_count: usize,
}

impl Graph {
    pub fn from_input(input: &GraphInput) -> Result<Self, InputError> {
        if input.vertices.len() > LIMIT_VERTICES {
            return Err(InputError::TooManyVertices(input.vertices.len()));
        }
        if input.edges.len() > LIMIT_RAW_EDGES {
            return Err(InputError::TooManyEdges(input.edges.len()));
        }

        let mut names = Vec::with_capacity(input.vertices.len());
        let mut index = BTreeMap::new();
        for name in &input.vertices {
            if name.is_empty() {
                return Err(InputError::EmptyVertexName);
            }
            if index.contains_key(name) {
                return Err(InputError::DuplicateVertexName(name.clone()));
            }
            index.insert(name.clone(), names.len());
            names.push(name.clone());
        }

        for edge in &input.edges {
            if !index.contains_key(&edge.u) {
                return Err(InputError::MissingVertex(edge.u.clone()));
            }
            if !index.contains_key(&edge.v) {
                return Err(InputError::MissingVertex(edge.v.clone()));
            }
            if edge.u == edge.v {
                return Err(InputError::SelfLoop(edge.u.clone()));
            }
        }

        let n = names.len();
        let mut adjacency = vec![BTreeSet::new(); n];
        let mut duplicate_edge_count = 0usize;

        for edge in &input.edges {
            let u = index[&edge.u];
            let v = index[&edge.v];
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            if !adjacency[a].insert(b) {
                duplicate_edge_count += 1;
            } else {
                adjacency[b].insert(a);
            }
        }

        let edge_count = adjacency
            .iter()
            .enumerate()
            .map(|(u, neighbors)| neighbors.range((u + 1)..).count())
            .sum();

        Ok(Self {
            names,
            index,
            adjacency,
            edge_count,
            duplicate_edge_count,
        })
    }

    #[cfg(test)]
    pub fn from_parts(names: Vec<String>, pairs: &[(usize, usize)]) -> Self {
        let mut index = BTreeMap::new();
        for (id, name) in names.iter().enumerate() {
            index.insert(name.clone(), id);
        }
        let mut adjacency = vec![BTreeSet::new(); names.len()];
        for &(raw_u, raw_v) in pairs {
            let (u, v) = if raw_u < raw_v {
                (raw_u, raw_v)
            } else {
                (raw_v, raw_u)
            };
            adjacency[u].insert(v);
            adjacency[v].insert(u);
        }
        let edge_count = adjacency
            .iter()
            .enumerate()
            .map(|(u, neighbors)| neighbors.range((u + 1)..).count())
            .sum();
        Self {
            names,
            index,
            adjacency,
            edge_count,
            duplicate_edge_count: 0,
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.names.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn duplicate_edge_count(&self) -> usize {
        self.duplicate_edge_count
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn id(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    pub fn name(&self, id: usize) -> &str {
        &self.names[id]
    }

    pub fn neighbors(&self, vertex: usize) -> &BTreeSet<usize> {
        &self.adjacency[vertex]
    }

    pub fn adjacent(&self, u: usize, v: usize) -> bool {
        self.adjacency[u].contains(&v)
    }

    pub fn components(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.vertex_count()];
        let mut components = Vec::new();
        for start in 0..self.vertex_count() {
            if seen[start] {
                continue;
            }
            let mut stack = vec![start];
            let mut component = Vec::new();
            seen[start] = true;
            while let Some(vertex) = stack.pop() {
                component.push(vertex);
                for neighbor in &self.adjacency[vertex] {
                    if !seen[*neighbor] {
                        seen[*neighbor] = true;
                        stack.push(*neighbor);
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
        components
    }

    pub fn connected_vertices(&self) -> bool {
        let n = self.vertex_count();
        if n <= 1 {
            return true;
        }
        let mut seen = vec![false; n];
        let mut stack = vec![0];
        seen[0] = true;
        let mut count = 1usize;
        while let Some(vertex) = stack.pop() {
            for neighbor in &self.adjacency[vertex] {
                if !seen[*neighbor] {
                    seen[*neighbor] = true;
                    count += 1;
                    stack.push(*neighbor);
                }
            }
        }
        count == n
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GraphSummary {
    pub vertex_count: usize,
    pub edge_count: usize,
    pub duplicate_edge_count: usize,
    pub connected: bool,
    pub components: Vec<Vec<String>>,
}

impl Graph {
    pub fn summary(&self) -> GraphSummary {
        let components = self
            .components()
            .into_iter()
            .map(|component| {
                component
                    .into_iter()
                    .map(|id| self.names[id].clone())
                    .collect()
            })
            .collect();
        GraphSummary {
            vertex_count: self.vertex_count(),
            edge_count: self.edge_count(),
            duplicate_edge_count: self.duplicate_edge_count,
            connected: self.connected_vertices(),
            components,
        }
    }
}
