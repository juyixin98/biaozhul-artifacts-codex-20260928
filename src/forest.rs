//! Forest helper used by the dynamic MSF core.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::model::{Edge, EdgeId, Vertex};

/// Undirected adjacency restricted to current spanning-tree edges.
#[derive(Debug, Clone, Default)]
pub struct Forest {
    adjacency: BTreeMap<Vertex, BTreeSet<(EdgeId, Vertex)>>,
    edge_ids: BTreeSet<EdgeId>,
}

impl Forest {
    /// Create an empty forest.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct from graph edges that are selected by `edge_ids`.
    pub fn from_selected<'a>(
        edges: impl Iterator<Item = &'a Edge>,
        edge_ids: &BTreeSet<EdgeId>,
    ) -> Self {
        let mut forest = Self::new();
        for edge in edges {
            if edge_ids.contains(&edge.id) {
                forest.add(edge);
            }
        }
        forest
    }

    /// Sorted selected edge IDs.
    pub fn edge_ids(&self) -> &BTreeSet<EdgeId> {
        &self.edge_ids
    }

    /// Returns true when the edge ID is currently selected.
    pub fn contains(&self, edge_id: &str) -> bool {
        self.edge_ids.contains(edge_id)
    }

    /// Add an edge. The caller guarantees it joins distinct forest components.
    pub fn add(&mut self, edge: &Edge) {
        self.edge_ids.insert(edge.id.clone());
        self.adjacency
            .entry(edge.u.clone())
            .or_default()
            .insert((edge.id.clone(), edge.v.clone()));
        self.adjacency
            .entry(edge.v.clone())
            .or_default()
            .insert((edge.id.clone(), edge.u.clone()));
    }

    /// Remove a selected edge.
    pub fn remove(&mut self, edge_id: &str, u: &str, v: &str) {
        self.edge_ids.remove(edge_id);
        if let Some(neighbors) = self.adjacency.get_mut(u) {
            neighbors.remove(&(edge_id.to_string(), v.to_string()));
        }
        if u != v {
            if let Some(neighbors) = self.adjacency.get_mut(v) {
                neighbors.remove(&(edge_id.to_string(), u.to_string()));
            }
        }
    }

    /// Returns the ordered forest path from `source` to `target`, if connected.
    pub fn path(&self, source: &str, target: &str) -> Option<Vec<EdgeId>> {
        if source == target {
            return Some(Vec::new());
        }

        let mut previous: BTreeMap<Vertex, (EdgeId, Vertex)> = BTreeMap::new();
        let mut queue = VecDeque::from([source.to_string()]);
        previous.insert(source.to_string(), (String::new(), String::new()));

        while let Some(vertex) = queue.pop_front() {
            let Some(neighbors) = self.adjacency.get(&vertex) else {
                continue;
            };
            for (edge_id, next) in neighbors {
                if previous.contains_key(next) {
                    continue;
                }
                previous.insert(next.clone(), (edge_id.clone(), vertex.clone()));
                if next == target {
                    let mut result = vec![edge_id.clone()];
                    let mut cursor = vertex.clone();
                    while cursor != source {
                        let (edge, parent) = previous
                            .get(&cursor)
                            .expect("visited vertex has parent")
                            .clone();
                        result.push(edge);
                        cursor = parent;
                    }
                    result.reverse();
                    return Some(result);
                }
                queue.push_back(next.clone());
            }
        }

        None
    }

    /// Vertex set reachable from `root` through selected forest edges.
    pub fn component_from(&self, root: &str) -> BTreeSet<Vertex> {
        let mut seen = BTreeSet::new();
        seen.insert(root.to_string());
        let mut queue = VecDeque::from([root.to_string()]);

        while let Some(vertex) = queue.pop_front() {
            let Some(neighbors) = self.adjacency.get(&vertex) else {
                continue;
            };
            for (_, next) in neighbors {
                if seen.insert(next.clone()) {
                    queue.push_back(next.clone());
                }
            }
        }

        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(id: &str, u: &str, v: &str) -> Edge {
        Edge {
            id: id.to_string(),
            u: u.to_string(),
            v: v.to_string(),
            weight: 1,
        }
    }

    #[test]
    fn finds_paths_and_split_components() {
        let mut forest = Forest::new();
        forest.add(&e("ab", "a", "b"));
        forest.add(&e("bc", "b", "c"));
        forest.add(&e("de", "d", "e"));

        assert_eq!(
            forest.path("a", "c"),
            Some(vec!["ab".to_string(), "bc".to_string()])
        );
        assert_eq!(forest.path("a", "d"), None);

        forest.remove("bc", "b", "c");
        assert_eq!(
            forest.component_from("a"),
            BTreeSet::from(["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            forest.component_from("c"),
            BTreeSet::from(["c".to_string()])
        );
    }
}
