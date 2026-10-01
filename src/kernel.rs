use crate::model::{Edge, Graph, ModelError};
use crate::store::Snapshot;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub enum MaintenanceOp {
    AddEdge(Edge),
    DeleteEdge { id: String },
    SetWeight { id: String, weight: i64 },
}

#[derive(Debug, Clone, Serialize)]
pub struct StepReport {
    pub index: usize,
    pub operation: &'static str,
    pub edge_id: String,
    pub action: String,
    pub before_weight: Option<i64>,
    pub after_weight: Option<i64>,
    pub forest_edge_added: Option<String>,
    pub forest_edge_removed: Option<String>,
    pub total_weight_delta: String,
}

#[derive(Debug, Clone)]
pub struct DynamicMsf {
    graph: Graph,
    forest: BTreeSet<String>,
    total_weight: i128,
    max_edges: usize,
    max_edge_id_length: usize,
}

impl DynamicMsf {
    pub fn new(max_edges: usize, max_edge_id_length: usize) -> Self {
        Self {
            graph: Graph::new(),
            forest: BTreeSet::new(),
            total_weight: 0,
            max_edges,
            max_edge_id_length,
        }
    }

    pub fn from_snapshot(snapshot: &Snapshot, max_edges: usize, max_edge_id_length: usize) -> Self {
        Self {
            graph: Graph {
                vertices: snapshot.vertices.clone(),
                edges: snapshot.edges.clone(),
            },
            forest: snapshot.forest_edge_ids.clone(),
            total_weight: snapshot.total_weight,
            max_edges,
            max_edge_id_length,
        }
    }

    pub fn snapshot(&self, graph_id: &str, version: u64) -> Snapshot {
        Snapshot {
            graph_id: graph_id.to_string(),
            version,
            vertices: self.graph.vertices.clone(),
            edges: self.graph.edges.clone(),
            forest_edge_ids: self.forest.clone(),
            total_weight: self.total_weight,
        }
    }

    pub fn apply(&mut self, index: usize, op: MaintenanceOp) -> Result<StepReport, ModelError> {
        let mut step = match op {
            MaintenanceOp::AddEdge(edge) => self.add_edge(edge)?,
            MaintenanceOp::DeleteEdge { id } => self.delete_edge(&id)?,
            MaintenanceOp::SetWeight { id, weight } => self.set_weight(&id, weight)?,
        };
        step.index = index;
        Ok(step)
    }

    fn add_edge(&mut self, edge: Edge) -> Result<StepReport, ModelError> {
        let initial_total = self.total_weight;
        let edge_id = edge.id.clone();
        let after_weight = Some(edge.weight);
        let mut action = "kept_out_of_forest".to_string();
        let mut added: Option<String> = None;
        let mut removed: Option<String> = None;

        self.graph
            .add_edge(edge.clone(), self.max_edges, self.max_edge_id_length)?;

        if edge.is_self_loop() {
            action = "ignored_self_loop".to_string();
        } else {
            let mut dsu = ForestDsu::new(&self.graph.vertices, &self.graph.edges, &self.forest);
            if dsu.different(edge.u, edge.v) {
                self.forest.insert(edge.id.clone());
                self.total_weight = checked_add(self.total_weight, edge.weight as i128)?;
                added = Some(edge.id.clone());
                action = "added_to_forest".to_string();
            } else if let Some(path) = find_path(
                &self.graph.vertices,
                &self.graph.edges,
                &self.forest,
                edge.u,
                edge.v,
            ) {
                let heaviest = path
                    .iter()
                    .map(|id| self.graph.edges.get(id).expect("path edge exists"))
                    .fold(
                        self.graph.edges.get(&path[0]).expect("path edge exists"),
                        |acc, candidate| {
                            if preferred(candidate, acc) {
                                acc
                            } else {
                                candidate
                            }
                        },
                    );

                if preferred(&edge, heaviest) {
                    self.forest.remove(&heaviest.id);
                    self.forest.insert(edge.id.clone());
                    self.total_weight = checked_add(
                        self.total_weight - heaviest.weight as i128,
                        edge.weight as i128,
                    )?;
                    removed = Some(heaviest.id.clone());
                    added = Some(edge.id.clone());
                    action = "replaced_heavier_cycle_edge".to_string();
                }
            }
        }

        Ok(StepReport {
            index: 0,
            operation: "add_edge",
            edge_id,
            action,
            before_weight: None,
            after_weight,
            forest_edge_added: added,
            forest_edge_removed: removed,
            total_weight_delta: (self.total_weight - initial_total).to_string(),
        })
    }

    fn delete_edge(&mut self, id: &str) -> Result<StepReport, ModelError> {
        let initial_total = self.total_weight;
        let edge = self
            .graph
            .edges
            .get(id)
            .ok_or_else(|| ModelError::EdgeNotFound(id.to_string()))?
            .clone();
        let before_weight = Some(edge.weight);
        let mut action = "deleted_non_tree_edge".to_string();
        let mut added: Option<String> = None;
        let mut removed: Option<String> = None;

        if self.forest.remove(id) {
            self.total_weight = checked_add(self.total_weight, -(edge.weight as i128))?;
            removed = Some(id.to_string());

            if !edge.is_self_loop() {
                let mut dsu = ForestDsu::new(&self.graph.vertices, &self.graph.edges, &self.forest);
                if let Some(replacement) = self.minimum_crossing(&mut dsu, Some(id)) {
                    self.forest.insert(replacement.id.clone());
                    self.total_weight = checked_add(self.total_weight, replacement.weight as i128)?;
                    added = Some(replacement.id.clone());
                    action = "deleted_tree_edge_and_reconnected".to_string();
                } else {
                    action = "deleted_tree_edge_bridge".to_string();
                }
            }
        }

        self.graph.remove_edge(id)?;

        Ok(StepReport {
            index: 0,
            operation: "delete_edge",
            edge_id: id.to_string(),
            action,
            before_weight,
            after_weight: None,
            forest_edge_added: added,
            forest_edge_removed: removed,
            total_weight_delta: (self.total_weight - initial_total).to_string(),
        })
    }

    fn set_weight(&mut self, id: &str, new_weight: i64) -> Result<StepReport, ModelError> {
        let initial_total = self.total_weight;
        let old = self.graph.set_weight(id, new_weight)?;
        let old_weight = old.weight;
        let before_weight = Some(old_weight);
        let after_weight = Some(new_weight);
        let mut action = "weight_changed_non_tree_edge".to_string();
        let mut added: Option<String> = None;
        let mut removed: Option<String> = None;

        if old.is_self_loop() {
            action = "weight_changed_self_loop".to_string();
        } else if self.forest.contains(id) {
            let mut trial_forest = self.forest.clone();
            trial_forest.remove(id);
            let mut dsu = ForestDsu::new(&self.graph.vertices, &self.graph.edges, &trial_forest);
            let updated = self.graph.edges.get(id).expect("updated edge exists");

            if let Some(alternative) = self.minimum_crossing(&mut dsu, Some(id)) {
                if preferred(&alternative, updated) {
                    self.forest.remove(id);
                    self.forest.insert(alternative.id.clone());
                    self.total_weight = checked_add(
                        self.total_weight - old_weight as i128,
                        alternative.weight as i128,
                    )?;
                    removed = Some(id.to_string());
                    added = Some(alternative.id.clone());
                    action = "tree_weight_increased_and_reconnected".to_string();
                } else {
                    self.total_weight =
                        checked_add(self.total_weight - old_weight as i128, new_weight as i128)?;
                    action = "tree_weight_changed_still_selected".to_string();
                }
            } else {
                self.total_weight =
                    checked_add(self.total_weight - old_weight as i128, new_weight as i128)?;
                action = "tree_weight_changed_still_selected".to_string();
            }
        } else {
            let updated = self.graph.edges.get(id).expect("updated edge exists");
            if let Some(path) = find_path(
                &self.graph.vertices,
                &self.graph.edges,
                &self.forest,
                old.u,
                old.v,
            ) {
                let heaviest = path
                    .iter()
                    .map(|path_id| self.graph.edges.get(path_id).expect("path edge exists"))
                    .fold(
                        self.graph.edges.get(&path[0]).expect("path edge exists"),
                        |acc, candidate| {
                            if preferred(candidate, acc) {
                                acc
                            } else {
                                candidate
                            }
                        },
                    );

                if preferred(updated, heaviest) {
                    self.forest.remove(&heaviest.id);
                    self.forest.insert(id.to_string());
                    self.total_weight = checked_add(
                        self.total_weight - heaviest.weight as i128,
                        new_weight as i128,
                    )?;
                    removed = Some(heaviest.id.clone());
                    added = Some(id.to_string());
                    action = "non_tree_weight_decreased_entered_forest".to_string();
                }
            }
        }

        Ok(StepReport {
            index: 0,
            operation: "set_weight",
            edge_id: id.to_string(),
            action,
            before_weight,
            after_weight,
            forest_edge_added: added,
            forest_edge_removed: removed,
            total_weight_delta: (self.total_weight - initial_total).to_string(),
        })
    }

    fn minimum_crossing(&self, dsu: &mut ForestDsu, excluded: Option<&str>) -> Option<Edge> {
        self.graph
            .edges
            .values()
            .filter(|edge| Some(edge.id.as_str()) != excluded)
            .filter(|edge| !edge.is_self_loop())
            .filter(|edge| dsu.different(edge.u, edge.v))
            .fold(None, |best: Option<Edge>, candidate| match best {
                None => Some(candidate.clone()),
                Some(best) => Some(if preferred(candidate, &best) {
                    candidate.clone()
                } else {
                    best
                }),
            })
    }
}

fn preferred(a: &Edge, b: &Edge) -> bool {
    (a.weight, a.id.as_str()) < (b.weight, b.id.as_str())
}

fn checked_add(a: i128, b: i128) -> Result<i128, ModelError> {
    a.checked_add(b).ok_or(ModelError::ArithmeticOverflow)
}

struct ForestDsu {
    index: BTreeMap<i64, usize>,
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl ForestDsu {
    fn new(
        vertices: &BTreeSet<i64>,
        edges: &BTreeMap<String, Edge>,
        forest: &BTreeSet<String>,
    ) -> Self {
        let index: BTreeMap<_, _> = vertices
            .iter()
            .enumerate()
            .map(|(index, value)| (*value, index))
            .collect();
        let count = vertices.len();
        let mut dsu = Self {
            index,
            parent: (0..count).collect(),
            rank: vec![0; count],
        };

        for id in forest {
            if let Some(edge) = edges.get(id) {
                if !edge.is_self_loop() {
                    dsu.union(edge.u, edge.v);
                }
            }
        }
        dsu
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: i64, b: i64) {
        let Some(&ai) = self.index.get(&a) else {
            return;
        };
        let Some(&bi) = self.index.get(&b) else {
            return;
        };
        let ar = self.find(ai);
        let br = self.find(bi);
        if ar == br {
            return;
        }
        match self.rank[ar].cmp(&self.rank[br]) {
            std::cmp::Ordering::Less => self.parent[ar] = br,
            std::cmp::Ordering::Greater => self.parent[br] = ar,
            std::cmp::Ordering::Equal => {
                self.parent[br] = ar;
                self.rank[ar] += 1;
            }
        }
    }

    fn different(&mut self, a: i64, b: i64) -> bool {
        match (self.index.get(&a), self.index.get(&b)) {
            (Some(&ai), Some(&bi)) => self.find(ai) != self.find(bi),
            _ => true,
        }
    }
}

fn find_path(
    vertices: &BTreeSet<i64>,
    edges: &BTreeMap<String, Edge>,
    forest: &BTreeSet<String>,
    source: i64,
    target: i64,
) -> Option<Vec<String>> {
    if source == target {
        return Some(Vec::new());
    }

    let mut adjacency: BTreeMap<i64, Vec<(i64, String)>> = vertices
        .iter()
        .map(|vertex| (*vertex, Vec::new()))
        .collect();

    for id in forest {
        let Some(edge) = edges.get(id) else { continue };
        if edge.is_self_loop() {
            continue;
        }
        adjacency
            .get_mut(&edge.u)
            .expect("adjacency exists")
            .push((edge.v, id.clone()));
        adjacency
            .get_mut(&edge.v)
            .expect("adjacency exists")
            .push((edge.u, id.clone()));
    }

    let mut previous: BTreeMap<i64, (i64, String)> = BTreeMap::new();
    let mut stack = vec![source];

    while let Some(vertex) = stack.pop() {
        if vertex == target {
            break;
        }
        for (next, edge_id) in adjacency.get(&vertex).cloned().unwrap_or_default() {
            if next == source || previous.contains_key(&next) {
                continue;
            }
            previous.insert(next, (vertex, edge_id));
            stack.push(next);
        }
    }

    if !previous.contains_key(&target) {
        return None;
    }

    let mut path = Vec::new();
    let mut current = target;
    while current != source {
        let (previous_vertex, edge_id) = previous
            .get(&current)
            .expect("target path predecessor exists")
            .clone();
        path.push(edge_id);
        current = previous_vertex;
    }
    path.reverse();
    Some(path)
}
