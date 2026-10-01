//! Incremental minimum-spanning-forest maintenance.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::forest::Forest;
use crate::model::{Edge, EdgeId, Graph, Vertex, Weight};
use crate::operations::Operation;

/// Failure category returned while preparing or applying a batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "category", rename_all = "snake_case")]
pub enum CoreError {
    /// Batch did not contain any operations.
    EmptyBatch,
    /// An operation referred to an edge ID that does not exist in that position.
    UnknownEdge {
        /// Zero-based operation position.
        operation_index: usize,
        /// Missing edge ID.
        edge_id: String,
    },
    /// Request data had invalid identifiers or endpoints.
    InvalidEdge {
        /// Zero-based operation position.
        operation_index: usize,
        /// Edge ID, if one was supplied.
        edge_id: Option<String>,
        /// Machine-readable reason.
        reason: String,
    },
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreError::EmptyBatch => write!(f, "batch is empty"),
            CoreError::UnknownEdge {
                operation_index,
                edge_id,
            } => write!(f, "operation {operation_index}: unknown edge '{edge_id}'"),
            CoreError::InvalidEdge {
                operation_index,
                edge_id,
                reason,
            } => write!(
                f,
                "operation {operation_index}: invalid edge {}: {reason}",
                edge_id.as_deref().unwrap_or("<missing>")
            ),
        }
    }
}

impl std::error::Error for CoreError {}

/// One explainable algorithm step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StepEvent {
    /// Zero-based operation position, or None for batch boundaries.
    pub operation_index: Option<usize>,
    /// Edge relevant to the step.
    pub edge_id: Option<String>,
    /// Stable machine-readable step name.
    pub step: String,
    /// Human-readable explanation with the decisive ranking or cut result.
    pub detail: String,
    /// Candidate considered for reconnecting or replacing a tree edge.
    pub candidate_edge_id: Option<String>,
    /// Tree edge removed by a swap.
    pub replaced_edge_id: Option<String>,
    /// Forest total immediately after this step.
    pub total_weight: i128,
}

/// Mutable graph together with its maintained minimum spanning forest.
#[derive(Debug, Clone)]
pub struct DynamicMsf {
    graph: Graph,
    forest: Forest,
    total: i128,
}

impl Default for DynamicMsf {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicMsf {
    /// Start from an empty versioned graph.
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            forest: Forest::new(),
            total: 0,
        }
    }

    /// Build canonical MSF state from a complete graph using Kruskal's algorithm.
    pub fn from_graph(mut graph: Graph) -> Self {
        let mut forest = Forest::new();
        let mut total: i128 = 0;
        let mut parent: BTreeMap<Vertex, Vertex> = graph
            .vertices()
            .map(|vertex| (vertex.clone(), vertex.clone()))
            .collect();

        let mut edges: Vec<&Edge> = graph.edges().filter(|edge| !edge.is_self_loop()).collect();
        edges.sort_by(|a, b| edge_rank(a).cmp(&edge_rank(b)));

        for edge in edges {
            if union(&mut parent, &edge.u, &edge.v) {
                forest.add(edge);
                total += i128::from(edge.weight);
            }
        }

        graph.set_version(0);
        Self {
            graph,
            forest,
            total,
        }
    }

    /// Complete graph version.
    pub fn version(&self) -> u64 {
        self.graph.version()
    }

    /// Current complete graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Sorted IDs of current forest edges.
    pub fn forest_ids(&self) -> &BTreeSet<EdgeId> {
        self.forest.edge_ids()
    }

    /// Exact forest total, using i128 for summed i64 weights.
    pub fn total_weight(&self) -> i128 {
        self.total
    }

    /// Validate and apply one complete batch, publishing the next graph version.
    pub fn apply_batch(&mut self, batch: &[Operation]) -> Result<Vec<StepEvent>, CoreError> {
        if batch.is_empty() {
            return Err(CoreError::EmptyBatch);
        }
        self.validate_batch(batch)?;

        let mut events = vec![StepEvent {
            operation_index: None,
            edge_id: None,
            step: "batch_started".to_string(),
            detail: format!(
                "processing {} operation(s) against complete graph version {}",
                batch.len(),
                self.version()
            ),
            candidate_edge_id: None,
            replaced_edge_id: None,
            total_weight: self.total,
        }];

        for (index, operation) in batch.iter().enumerate() {
            match operation {
                Operation::UpsertEdge(edge) => self.apply_upsert(index, edge, &mut events),
                Operation::UpdateWeight { id, weight } => {
                    self.apply_weight_update(index, id, *weight, &mut events);
                }
                Operation::DeleteEdge { id } => self.apply_delete(index, id, &mut events),
            }
        }

        let next_version = self.version().saturating_add(1);
        self.graph.set_version(next_version);
        events.push(StepEvent {
            operation_index: None,
            edge_id: None,
            step: "batch_committed".to_string(),
            detail: format!("published complete graph version {next_version}"),
            candidate_edge_id: None,
            replaced_edge_id: None,
            total_weight: self.total,
        });

        Ok(events)
    }

    fn validate_batch(&self, batch: &[Operation]) -> Result<(), CoreError> {
        let mut simulated_existence: BTreeSet<EdgeId> =
            self.graph.edges().map(|edge| edge.id.clone()).collect();

        for (index, operation) in batch.iter().enumerate() {
            match operation {
                Operation::UpsertEdge(edge) => {
                    if edge.id.trim().is_empty() {
                        return Err(CoreError::InvalidEdge {
                            operation_index: index,
                            edge_id: Some(edge.id.clone()),
                            reason: "edge id must not be empty".to_string(),
                        });
                    }
                    if edge.u.trim().is_empty() || edge.v.trim().is_empty() {
                        return Err(CoreError::InvalidEdge {
                            operation_index: index,
                            edge_id: Some(edge.id.clone()),
                            reason: "edge endpoints must not be empty".to_string(),
                        });
                    }
                    simulated_existence.insert(edge.id.clone());
                }
                Operation::UpdateWeight { id, .. } | Operation::DeleteEdge { id } => {
                    if id.trim().is_empty() {
                        return Err(CoreError::InvalidEdge {
                            operation_index: index,
                            edge_id: Some(id.clone()),
                            reason: "edge id must not be empty".to_string(),
                        });
                    }
                    if !simulated_existence.contains(id) {
                        return Err(CoreError::UnknownEdge {
                            operation_index: index,
                            edge_id: id.clone(),
                        });
                    }
                    if matches!(operation, Operation::DeleteEdge { .. }) {
                        simulated_existence.remove(id);
                    }
                }
            }
        }

        Ok(())
    }
    fn apply_upsert(&mut self, index: usize, edge: &Edge, events: &mut Vec<StepEvent>) {
        if let Some(old) = self.graph.edge(&edge.id).cloned() {
            let was_tree = self.forest.contains(&old.id);
            if was_tree {
                self.forest.remove(&old.id, &old.u, &old.v);
                self.total -= i128::from(old.weight);
            }
            self.graph.remove_edge(&old.id);
            self.graph.upsert_edge(edge.clone());

            events.push(self.event(
                index,
                edge.id.clone(),
                "edge_replaced".to_string(),
                format!(
                    "replaced existing edge ({}, {}, w={}); previous forest membership was {}",
                    old.u,
                    old.v,
                    old.weight,
                    if was_tree { "selected" } else { "not selected" }
                ),
                None,
                if was_tree { Some(old.id.clone()) } else { None },
            ));

            if edge.is_self_loop() {
                self.record_self_loop_ignored(index, edge, events);
            }
            if was_tree {
                self.reconnect_after_tree_removal(
                    index,
                    &old.id,
                    &old.u,
                    &old.v,
                    Some(edge.id.clone()),
                    events,
                );
            }
            if !edge.is_self_loop() && !self.forest.contains(&edge.id) {
                self.consider_non_tree_edge(index, edge, events);
            }
        } else {
            self.graph.upsert_edge(edge.clone());
            if edge.is_self_loop() {
                self.record_self_loop_ignored(index, edge, events);
            } else {
                self.consider_non_tree_edge(index, edge, events);
            }
        }
    }

    fn apply_weight_update(
        &mut self,
        index: usize,
        id: &str,
        new_weight: Weight,
        events: &mut Vec<StepEvent>,
    ) {
        let old = self
            .graph
            .edge(id)
            .cloned()
            .expect("batch validation guarantees edge");

        if old.weight == new_weight {
            events.push(self.event(
                index,
                id.to_string(),
                "weight_update_noop".to_string(),
                format!("weight already equals {new_weight}"),
                None,
                None,
            ));
            return;
        }

        if old.is_self_loop() {
            self.graph.upsert_edge(Edge {
                weight: new_weight,
                ..old.clone()
            });
            events.push(self.event(
                index,
                id.to_string(),
                "self_loop_weight_updated".to_string(),
                format!(
                    "updated ignored self-loop weight {} -> {new_weight}; forest unchanged",
                    old.weight
                ),
                None,
                None,
            ));
            return;
        }

        let is_tree = self.forest.contains(id);
        let updated = Edge {
            weight: new_weight,
            ..old.clone()
        };
        self.graph.upsert_edge(updated.clone());

        if is_tree {
            let old_weight = old.weight;
            self.total -= i128::from(old_weight);
            self.forest.remove(id, &updated.u, &updated.v);
            let replacement = self.minimum_edge_between_split(&updated.u, &updated.v, None);

            events.push(self.event(
                index,
                id.to_string(),
                "tree_edge_weight_updated".to_string(),
                format!("updated selected edge weight {old_weight} -> {new_weight}; checked its fundamental cut"),
                replacement.as_ref().map(|edge| edge.id.clone()),
                None,
            ));

            if let Some(candidate) = replacement {
                if edge_rank(&candidate) < edge_rank(&updated) {
                    self.forest.add(&candidate);
                    self.total += i128::from(candidate.weight);
                    events.push(self.event(
                        index,
                        id.to_string(),
                        "tree_edge_replaced_after_increase".to_string(),
                        format!(
                            "minimum cut edge {} (w={}) is better than updated edge rank ({}, {})",
                            candidate.id, candidate.weight, new_weight, id
                        ),
                        Some(candidate.id.clone()),
                        Some(id.to_string()),
                    ));
                } else {
                    self.forest.add(&updated);
                    self.total += i128::from(new_weight);
                    events.push(self.event(
                        index,
                        id.to_string(),
                        "tree_edge_retained".to_string(),
                        format!(
                            "updated edge rank ({}, {}) is no worse than best cut candidate ({}, {})",
                            new_weight, id, candidate.weight, candidate.id
                        ),
                        Some(candidate.id.clone()),
                        None,
                    ));
                }
            } else {
                self.forest.add(&updated);
                self.total += i128::from(new_weight);
                events.push(self.event(
                    index,
                    id.to_string(),
                    "tree_edge_retained".to_string(),
                    "updated selected edge remains the only edge across its cut".to_string(),
                    None,
                    None,
                ));
            }
        } else {
            events.push(self.event(
                index,
                id.to_string(),
                "non_tree_edge_weight_updated".to_string(),
                format!(
                    "updated non-tree edge weight {} -> {new_weight}; checking tree path",
                    old.weight
                ),
                None,
                None,
            ));
            if !self.forest.contains(&updated.id) {
                self.consider_non_tree_edge(index, &updated, events);
            }
        }
    }

    fn apply_delete(&mut self, index: usize, id: &str, events: &mut Vec<StepEvent>) {
        let old = self
            .graph
            .remove_edge(id)
            .expect("batch validation guarantees edge");

        if old.is_self_loop() {
            events.push(self.event(
                index,
                id.to_string(),
                "self_loop_deleted".to_string(),
                "deleted ignored self-loop; forest unchanged".to_string(),
                None,
                None,
            ));
            return;
        }

        if !self.forest.contains(id) {
            events.push(self.event(
                index,
                id.to_string(),
                "non_tree_edge_deleted".to_string(),
                "deleted non-tree edge; forest unchanged".to_string(),
                None,
                None,
            ));
            return;
        }

        self.forest.remove(id, &old.u, &old.v);
        self.total -= i128::from(old.weight);
        events.push(self.event(
            index,
            id.to_string(),
            "tree_edge_deleted".to_string(),
            format!(
                "removed selected edge ({}, {}, w={}); scanning complete graph for minimum cut edge",
                old.u, old.v, old.weight
            ),
            None,
            Some(id.to_string()),
        ));
        self.reconnect_after_tree_removal(index, id, &old.u, &old.v, None, events);
    }

    fn consider_non_tree_edge(&mut self, index: usize, edge: &Edge, events: &mut Vec<StepEvent>) {
        let path = match self.forest.path(&edge.u, &edge.v) {
            Some(path) => path,
            None => {
                self.forest.add(edge);
                self.total += i128::from(edge.weight);
                events.push(self.event(
                    index,
                    edge.id.clone(),
                    "edge_connected_components".to_string(),
                    format!(
                        "new edge joins distinct components; selected with weight {}",
                        edge.weight
                    ),
                    None,
                    None,
                ));
                return;
            }
        };

        let maximum = path
            .iter()
            .filter(|id| id.as_str() != edge.id)
            .map(|id| {
                self.graph
                    .edge(id)
                    .expect("path edges exist in complete graph")
                    .clone()
            })
            .max_by(|a, b| edge_rank(a).cmp(&edge_rank(b)))
            .expect("cycle path is nonempty");

        if edge_rank(edge) < edge_rank(&maximum) {
            self.forest.remove(&maximum.id, &maximum.u, &maximum.v);
            self.forest.add(edge);
            self.total += i128::from(edge.weight) - i128::from(maximum.weight);
            events.push(self.event(
                index,
                edge.id.clone(),
                "cycle_max_edge_replaced".to_string(),
                format!(
                    "edge rank ({}, {}) replaces larger path edge {} (w={})",
                    edge.weight, edge.id, maximum.id, maximum.weight
                ),
                Some(maximum.id.clone()),
                Some(maximum.id.clone()),
            ));
        } else {
            events.push(self.event(
                index,
                edge.id.clone(),
                "cycle_edge_not_selected".to_string(),
                format!(
                    "edge rank ({}, {}) is no better than maximum path edge {} ({}, {})",
                    edge.weight, edge.id, maximum.id, maximum.weight, maximum.id
                ),
                Some(maximum.id.clone()),
                None,
            ));
        }
    }

    fn reconnect_after_tree_removal(
        &mut self,
        index: usize,
        removed_id: &str,
        left_root: &str,
        right_root: &str,
        excluded_edge_id: Option<EdgeId>,
        events: &mut Vec<StepEvent>,
    ) {
        let replacement = self.minimum_edge_between_split(left_root, right_root, excluded_edge_id);
        match replacement {
            Some(edge) => {
                self.forest.add(&edge);
                self.total += i128::from(edge.weight);
                events.push(self.event(
                    index,
                    removed_id.to_string(),
                    "minimum_cut_edge_selected".to_string(),
                    format!(
                        "selected minimum cross-cut replacement {} ({}, {}, w={})",
                        edge.id, edge.u, edge.v, edge.weight
                    ),
                    Some(edge.id.clone()),
                    Some(removed_id.to_string()),
                ));
            }
            None => events.push(self.event(
                index,
                removed_id.to_string(),
                "bridge_removal_split_component".to_string(),
                "no edge crosses the cut; components remain separate in the forest".to_string(),
                None,
                Some(removed_id.to_string()),
            )),
        }
    }

    fn minimum_edge_between_split(
        &self,
        left_root: &str,
        right_root: &str,
        excluded_edge_id: Option<EdgeId>,
    ) -> Option<Edge> {
        let left_component = self.forest.component_from(left_root);
        let right_component = self.forest.component_from(right_root);
        self.graph
            .edges()
            .filter(|edge| {
                !edge.is_self_loop()
                    && excluded_edge_id.as_ref() != Some(&edge.id)
                    && !self.forest.contains(&edge.id)
                    && ((left_component.contains(&edge.u) && right_component.contains(&edge.v))
                        || (left_component.contains(&edge.v) && right_component.contains(&edge.u)))
            })
            .min_by(|a, b| edge_rank(a).cmp(&edge_rank(b)))
            .cloned()
    }

    fn record_self_loop_ignored(&mut self, index: usize, edge: &Edge, events: &mut Vec<StepEvent>) {
        events.push(self.event(
            index,
            edge.id.clone(),
            "self_loop_ignored".to_string(),
            format!(
                "self-loop on vertex {} is present in graph state but excluded from forest",
                edge.u
            ),
            None,
            None,
        ));
    }

    fn event(
        &self,
        operation_index: usize,
        edge_id: EdgeId,
        step: String,
        detail: String,
        candidate_edge_id: Option<EdgeId>,
        replaced_edge_id: Option<EdgeId>,
    ) -> StepEvent {
        StepEvent {
            operation_index: Some(operation_index),
            edge_id: Some(edge_id),
            step,
            detail,
            candidate_edge_id,
            replaced_edge_id,
            total_weight: self.total,
        }
    }
}

fn edge_rank(edge: &Edge) -> (Weight, &str) {
    (edge.weight, edge.id.as_str())
}

fn find(parent: &mut BTreeMap<Vertex, Vertex>, vertex: &Vertex) -> Vertex {
    let mut current = vertex.clone();
    while let Some(parent_vertex) = parent.get(&current) {
        if *parent_vertex == current {
            return current;
        }
        current = parent_vertex.clone();
    }
    parent.insert(vertex.clone(), vertex.clone());
    vertex.clone()
}

fn union(parent: &mut BTreeMap<Vertex, Vertex>, left: &str, right: &str) -> bool {
    let left_root = find(parent, &left.to_string());
    let right_root = find(parent, &right.to_string());
    if left_root == right_root {
        return false;
    }
    parent.insert(right_root.clone(), left_root.clone());
    true
}
