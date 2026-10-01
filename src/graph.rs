use std::collections::{HashMap, HashSet};

pub const MAX_VERTICES: usize = 128;
pub const MAX_EDGE_RECORDS: usize = 20_000;
pub const MAX_VERTEX_LABEL_CHARS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    InputError(String),
    StateConflict(String),
    ResourceExhausted(String),
}

impl BuildError {
    pub fn input(message: impl Into<String>) -> Self {
        Self::InputError(message.into())
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::StateConflict(message.into())
    }

    pub fn exhausted(message: impl Into<String>) -> Self {
        Self::ResourceExhausted(message.into())
    }
}

#[derive(Debug, Clone)]
pub struct Graph {
    vertices: Vec<String>,
    index: HashMap<String, usize>,
    adjacency: Vec<Vec<usize>>,
    edge_count: usize,
    duplicate_edge_count: usize,
}

impl Graph {
    pub fn from_parts(
        declared_vertices: Option<Vec<String>>,
        edges: Vec<(String, String)>,
    ) -> Result<Self, BuildError> {
        if edges.len() > MAX_EDGE_RECORDS {
            return Err(BuildError::exhausted(format!(
                "edge record count {} exceeds limit {}",
                edges.len(),
                MAX_EDGE_RECORDS
            )));
        }

        let mut vertices: Vec<String> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        if let Some(declared) = declared_vertices {
            for vertex in declared {
                validate_label(&vertex)?;
                if index.contains_key(&vertex) {
                    return Err(BuildError::conflict(format!(
                        "duplicate vertex declaration: {vertex}"
                    )));
                }
                let id = vertices.len();
                vertices.push(vertex.clone());
                index.insert(vertex, id);
            }
        }

        if vertices.len() > MAX_VERTICES {
            return Err(BuildError::exhausted(format!(
                "vertex count {} exceeds limit {}",
                vertices.len(),
                MAX_VERTICES
            )));
        }

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); vertices.len()];
        let mut unique_edges: HashSet<(usize, usize)> = HashSet::new();
        let mut duplicate_edge_count = 0usize;

        for (left, right) in edges {
            validate_label(&left)?;
            validate_label(&right)?;
            if left == right {
                return Err(BuildError::input(format!(
                    "self-loops are not supported in simple graphs: {left}"
                )));
            }

            let left_id = resolve_or_create(&left, &mut vertices, &mut index, &mut adjacency)?;
            let right_id = resolve_or_create(&right, &mut vertices, &mut index, &mut adjacency)?;

            let (u, v) = if left_id <= right_id {
                (left_id, right_id)
            } else {
                (right_id, left_id)
            };

            if unique_edges.insert((u, v)) {
                if unique_edges.len() > MAX_VERTICES * MAX_VERTICES / 2 {
                    return Err(BuildError::exhausted(format!(
                        "normalized edge count exceeds maximum for {MAX_VERTICES} vertices"
                    )));
                }
                adjacency[u].push(v);
                adjacency[v].push(u);
            } else {
                duplicate_edge_count += 1;
            }
        }

        for neighbors in adjacency.iter_mut() {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let edge_count = unique_edges.len();
        Ok(Self {
            vertices,
            index,
            adjacency,
            edge_count,
            duplicate_edge_count,
        })
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn duplicate_edge_count(&self) -> usize {
        self.duplicate_edge_count
    }

    pub fn vertex_id(&self, label: &str) -> Option<usize> {
        self.index.get(label).copied()
    }

    pub fn label(&self, id: usize) -> &str {
        &self.vertices[id]
    }

    pub fn labels(&self) -> &[String] {
        &self.vertices
    }

    pub fn neighbors(&self, vertex: usize) -> &[usize] {
        &self.adjacency[vertex]
    }

    pub fn degree(&self, vertex: usize) -> usize {
        self.adjacency[vertex].len()
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.adjacency[u].binary_search(&v).is_ok()
    }

    pub fn normalized_edges(&self) -> Vec<(String, String)> {
        self.normalized_edges_indexed()
            .into_iter()
            .map(|(u, v)| (self.vertices[u].clone(), self.vertices[v].clone()))
            .collect()
    }

    pub fn normalized_edges_indexed(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::with_capacity(self.edge_count);
        for (u, neighbors) in self.adjacency.iter().enumerate() {
            for &v in neighbors {
                if u < v {
                    edges.push((u, v));
                }
            }
        }
        edges
    }
}

fn validate_label(label: &str) -> Result<(), BuildError> {
    if label.is_empty() {
        return Err(BuildError::input("vertex labels must not be empty"));
    }
    let length = label.chars().count();
    if length > MAX_VERTEX_LABEL_CHARS {
        return Err(BuildError::input(format!(
            "vertex label is {length} characters; limit is {MAX_VERTEX_LABEL_CHARS}"
        )));
    }
    Ok(())
}

fn resolve_or_create(
    label: &str,
    vertices: &mut Vec<String>,
    index: &mut HashMap<String, usize>,
    adjacency: &mut Vec<Vec<usize>>,
) -> Result<usize, BuildError> {
    if let Some(id) = index.get(label) {
        return Ok(*id);
    }
    if vertices.len() >= MAX_VERTICES {
        return Err(BuildError::exhausted(format!(
            "vertex count would exceed limit {MAX_VERTICES}"
        )));
    }
    let id = vertices.len();
    vertices.push(label.to_owned());
    index.insert(label.to_owned(), id);
    adjacency.push(Vec::new());
    Ok(id)
}
