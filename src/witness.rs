use crate::store::Snapshot;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WitnessFailureCategory {
    UnknownEdge,
    DuplicateForestEdge,
    ForestEdgeIsSelfLoop,
    ForestCycle,
    NotSpanning,
    WrongWeight,
    UnstableOrNonMinimum,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessFailure {
    pub category: WitnessFailureCategory,
    pub message: String,
}

impl WitnessFailure {
    fn new(category: WitnessFailureCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessReport {
    pub status: &'static str,
    pub expected_total_weight: String,
    pub actual_total_weight: String,
    pub expected_forest_edge_ids: Vec<String>,
    pub actual_forest_edge_ids: Vec<String>,
    pub failures: Vec<WitnessFailure>,
    pub uncertainties: Vec<String>,
}

impl WitnessReport {
    fn fail(mut self, failure: WitnessFailure) -> Self {
        self.status = "rejected";
        self.failures.push(failure);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct WitnessClaimInput {
    #[serde(default)]
    pub vertices: BTreeSet<i64>,
    pub edges: BTreeMap<String, WitnessEdge>,
    #[serde(default)]
    pub forest_edge_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct WitnessEdge {
    pub u: i64,
    pub v: i64,
    pub weight: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReferenceResult {
    forest: BTreeSet<String>,
    total_weight: i128,
}

struct Dsu {
    index: BTreeMap<i64, usize>,
    parent: Vec<usize>,
}

impl Dsu {
    fn new(vertices: &BTreeSet<i64>) -> Self {
        let index = vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (*vertex, index))
            .collect();
        Self {
            index,
            parent: (0..vertices.len()).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: i64, b: i64) -> bool {
        match (self.index.get(&a), self.index.get(&b)) {
            (Some(&ai), Some(&bi)) => {
                let ar = self.find(ai);
                let br = self.find(bi);
                if ar == br {
                    false
                } else {
                    self.parent[br] = ar;
                    true
                }
            }
            _ => false,
        }
    }
}

pub fn verify_snapshot(snapshot: &Snapshot) -> WitnessReport {
    let input = WitnessClaimInput {
        vertices: snapshot.vertices.clone(),
        edges: snapshot
            .edges
            .iter()
            .map(|(id, edge)| {
                (
                    id.clone(),
                    WitnessEdge {
                        u: edge.u,
                        v: edge.v,
                        weight: edge.weight,
                    },
                )
            })
            .collect(),
        forest_edge_ids: snapshot.forest_edge_ids.iter().cloned().collect(),
    };
    verify_claim(&input, Some(snapshot.total_weight))
}

pub fn verify_claim(input: &WitnessClaimInput, claimed_total: Option<i128>) -> WitnessReport {
    let mut vertices = input.vertices.clone();
    for edge in input.edges.values() {
        vertices.insert(edge.u);
        vertices.insert(edge.v);
    }

    let actual_forest_edge_ids = input.forest_edge_ids.clone();
    let mut report = WitnessReport {
        status: "accepted",
        expected_total_weight: "0".to_string(),
        actual_total_weight: claimed_total
            .map(|value| value.to_string())
            .unwrap_or_default(),
        expected_forest_edge_ids: Vec::new(),
        actual_forest_edge_ids,
        uncertainties: Vec::new(),
        failures: Vec::new(),
    };

    let mut forest_ids: BTreeSet<&str> = BTreeSet::new();
    let mut actual_weight: i128 = 0;
    for id in &input.forest_edge_ids {
        if !forest_ids.insert(id.as_str()) {
            report = report.fail(WitnessFailure::new(
                WitnessFailureCategory::DuplicateForestEdge,
                format!("forest edge id `{id}` occurs more than once"),
            ));
            continue;
        }
        let Some(edge) = input.edges.get(id) else {
            report = report.fail(WitnessFailure::new(
                WitnessFailureCategory::UnknownEdge,
                format!("forest edge id `{id}` is not present in the full edge set"),
            ));
            continue;
        };
        if edge.u == edge.v {
            report = report.fail(WitnessFailure::new(
                WitnessFailureCategory::ForestEdgeIsSelfLoop,
                format!("forest edge `{id}` is a self-loop"),
            ));
            continue;
        }
        actual_weight = match actual_weight.checked_add(edge.weight as i128) {
            Some(value) => value,
            None => {
                report.uncertainties.push(
                    "claimed forest total weight overflowed i128 while witness was summing it"
                        .to_string(),
                );
                i128::MAX
            }
        };
    }

    // A separate DSU checks the claimed forest for acyclicity.
    let mut forest_dsu = Dsu::new(&vertices);
    for id in &input.forest_edge_ids {
        if let Some(edge) = input.edges.get(id) {
            if edge.u != edge.v && !forest_dsu.union(edge.u, edge.v) {
                report = report.fail(WitnessFailure::new(
                    WitnessFailureCategory::ForestCycle,
                    format!("forest edge `{id}` closes a cycle"),
                ));
            }
        }
    }

    // Independent Kruskal: sorted by (weight, stable edge id), not by kernel state.
    let mut ordered: Vec<_> = input.edges.iter().collect();
    ordered
        .sort_by(|(id_a, a), (id_b, b)| (a.weight, id_a.as_str()).cmp(&(b.weight, id_b.as_str())));

    let mut kruskal = Dsu::new(&vertices);
    let mut reference = ReferenceResult {
        forest: BTreeSet::new(),
        total_weight: 0,
    };
    for (id, edge) in ordered {
        if edge.u == edge.v {
            continue;
        }
        if kruskal.union(edge.u, edge.v) {
            reference.forest.insert(id.clone());
            reference.total_weight = reference
                .total_weight
                .checked_add(edge.weight as i128)
                .unwrap_or_else(|| i128::MAX);
        }
    }

    report.expected_total_weight = reference.total_weight.to_string();
    report.expected_forest_edge_ids = reference.forest.iter().cloned().collect();

    let non_isolated_edges: BTreeSet<i64> = input
        .edges
        .values()
        .flat_map(|edge| [edge.u, edge.v])
        .collect();
    let expected_component_count = vertices.len().saturating_sub(reference.forest.len());
    let claimed_component_count = vertices.len().saturating_sub(input.forest_edge_ids.len());
    if claimed_component_count != expected_component_count
        && non_isolated_edges
            .iter()
            .all(|vertex| vertices.contains(vertex))
    {
        report = report.fail(WitnessFailure::new(
            WitnessFailureCategory::NotSpanning,
            format!(
                "claimed forest has {claimed_component_count} components but the minimum spanning forest has {expected_component_count}"
            ),
        ));
    }

    if let Some(claimed) = claimed_total {
        if claimed != actual_weight || claimed != reference.total_weight {
            report = report.fail(WitnessFailure::new(
                WitnessFailureCategory::WrongWeight,
                format!(
                    "claimed total weight {claimed}, actual forest sum {actual_weight}, Kruskal total {}",
                    reference.total_weight
                ),
            ));
        }
    }

    let claimed_set: BTreeSet<&str> = input.forest_edge_ids.iter().map(String::as_str).collect();
    let claimed_set_owned: BTreeSet<String> = claimed_set.iter().map(|id| id.to_string()).collect();
    if claimed_set_owned != reference.forest {
        report = report.fail(WitnessFailure::new(
            WitnessFailureCategory::UnstableOrNonMinimum,
            format!(
                "claimed forest {claimed_set:?} differs from stable Kruskal forest {reference_forest:?}; equal weights select the lexicographically smaller edge ID",
                reference_forest = reference.forest
            ),
        ));
    }

    report
}
