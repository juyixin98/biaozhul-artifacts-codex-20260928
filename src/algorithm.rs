use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::{json, Value};

use crate::graph::Graph;
use crate::witness::{
    clique_forest_check, coverage_check, find_induced_cycle, maximal_cliques_check, peo_violation,
    CheckReport, ChordlessCycle, PeoViolation,
};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct McsStep {
    pub search_position: usize,
    pub selected_id: usize,
    pub selected_vertex: String,
    pub cardinalities_before: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CliqueNode {
    pub index: usize,
    pub vertex_ids: Vec<usize>,
    pub vertices: Vec<String>,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CliqueForestEdge {
    pub from: usize,
    pub to: usize,
    pub separator_ids: Vec<usize>,
    pub separator: Vec<String>,
    pub weight: usize,
}

#[derive(Debug, Clone, Serialize, Default, PartialEq, Eq)]
pub struct CliqueForest {
    pub nodes: Vec<CliqueNode>,
    pub edges: Vec<CliqueForestEdge>,
    pub component_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AlgorithmChecks {
    pub peo: CheckReport,
    pub maximal_cliques: CheckReport,
    pub clique_cover: CheckReport,
    pub running_intersection: CheckReport,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AnalysisResult {
    pub chordal: bool,
    pub graph: crate::graph::GraphSummary,
    pub mcs_search_order_ids: Vec<usize>,
    pub mcs_search_order: Vec<String>,
    pub elimination_order_ids: Vec<usize>,
    pub elimination_order: Vec<String>,
    pub mcs_steps: Vec<McsStep>,
    pub chordless_cycle: Option<ChordlessCycle>,
    pub peo_violation: Option<PeoViolation>,
    pub max_clique_size: usize,
    pub maximal_cliques: Vec<Vec<String>>,
    pub maximal_clique_ids: Vec<Vec<usize>>,
    pub clique_forest: CliqueForest,
    pub checks: Option<AlgorithmChecks>,
    pub intermediate: Value,
}

pub fn analyze(graph: &Graph) -> AnalysisResult {
    let (search_order, mcs_steps) = maximum_cardinality_search(graph);
    let elimination_order: Vec<usize> = search_order.iter().rev().copied().collect();
    let violation = peo_violation(graph, &elimination_order);
    let chordal = violation.is_none();

    let mut chordless_cycle = None;
    let mut maximal_clique_ids = Vec::new();
    let mut max_clique_size = 0usize;
    let mut clique_forest = CliqueForest::default();
    let mut checks = None;

    if chordal {
        maximal_clique_ids = maximal_cliques_from_peo(graph, &elimination_order);
        max_clique_size = maximal_clique_ids
            .iter()
            .map(|clique| clique.len())
            .max()
            .unwrap_or(0);
        clique_forest = build_max_weight_clique_forest(graph, &maximal_clique_ids);

        let peo_report = CheckReport {
            passed: true,
            failures: vec![],
            details: json!({
                "reason": "every vertex's later neighbors form a clique in the reversed MCS order"
            }),
        };
        let maximal_report = maximal_cliques_check(graph, &maximal_clique_ids);
        let coverage_report = coverage_check(graph, &maximal_clique_ids);
        let forest_edge_ids: Vec<(usize, usize)> = clique_forest
            .edges
            .iter()
            .map(|edge| (edge.from, edge.to))
            .collect();
        let rip_report = clique_forest_check(graph, &maximal_clique_ids, &forest_edge_ids);

        checks = Some(AlgorithmChecks {
            peo: peo_report,
            maximal_cliques: maximal_report,
            clique_cover: coverage_report,
            running_intersection: rip_report,
        });
    } else {
        chordless_cycle = find_induced_cycle(graph);
    }

    let components = graph.components();
    let intermediate = json!({
        "mcs_tie_break": "lowest internal id; internal id follows the request vertex order",
        "search_order_length": search_order.len(),
        "component_count": components.len(),
        "component_sizes": components.iter().map(|component| component.len()).collect::<Vec<_>>(),
        "normalized_edges": graph.edge_count(),
        "duplicate_edges_removed": graph.duplicate_edge_count(),
        "peo_check": if chordal { "passed" } else { "failed" },
        "cycle_detection": if chordal {
            "not_needed"
        } else if chordless_cycle.is_some() {
            "witness_found"
        } else {
            "no_witness_should_not_happen"
        }
    });

    AnalysisResult {
        chordal,
        graph: graph.summary(),
        mcs_search_order_ids: search_order.clone(),
        mcs_search_order: search_order
            .iter()
            .map(|id| graph.name(*id).to_string())
            .collect(),
        elimination_order_ids: elimination_order.clone(),
        elimination_order: elimination_order
            .iter()
            .map(|id| graph.name(*id).to_string())
            .collect(),
        mcs_steps,
        chordless_cycle,
        peo_violation: violation,
        max_clique_size,
        maximal_cliques: maximal_clique_ids
            .iter()
            .map(|clique| {
                clique
                    .iter()
                    .map(|id| graph.name(*id).to_string())
                    .collect()
            })
            .collect(),
        maximal_clique_ids,
        clique_forest,
        checks,
        intermediate,
    }
}

fn maximum_cardinality_search(graph: &Graph) -> (Vec<usize>, Vec<McsStep>) {
    let n = graph.vertex_count();
    let mut selected = vec![false; n];
    let mut cardinality = vec![0usize; n];
    let mut order = Vec::with_capacity(n);
    let mut steps = Vec::with_capacity(n);

    for search_position in 0..n {
        let mut chosen = None;
        for vertex in 0..n {
            if selected[vertex] {
                continue;
            }
            match chosen {
                None => chosen = Some(vertex),
                Some(current) => {
                    if cardinality[vertex] > cardinality[current]
                        || (cardinality[vertex] == cardinality[current] && vertex < current)
                    {
                        chosen = Some(vertex);
                    }
                }
            }
        }

        let vertex = chosen.expect("MCS must select a vertex until all vertices are selected");
        steps.push(McsStep {
            search_position,
            selected_id: vertex,
            selected_vertex: graph.name(vertex).to_string(),
            cardinalities_before: cardinality.clone(),
        });
        selected[vertex] = true;
        order.push(vertex);
        for neighbor in graph.neighbors(vertex) {
            if !selected[*neighbor] {
                cardinality[*neighbor] += 1;
            }
        }
    }
    (order, steps)
}

fn maximal_cliques_from_peo(graph: &Graph, elimination_order: &[usize]) -> Vec<Vec<usize>> {
    let n = graph.vertex_count();
    let mut position = vec![0usize; n];
    for (index, vertex) in elimination_order.iter().enumerate() {
        position[*vertex] = index;
    }

    let mut candidates: Vec<BTreeSet<usize>> = Vec::new();
    for vertex in elimination_order {
        let later_neighbors: BTreeSet<usize> = graph
            .neighbors(*vertex)
            .iter()
            .copied()
            .filter(|neighbor| position[*neighbor] > position[*vertex])
            .collect();
        let mut clique = later_neighbors;
        clique.insert(*vertex);
        candidates.push(clique);
    }

    let mut maximal = Vec::new();
    for index in 0..candidates.len() {
        let is_subset_of_other = candidates.iter().enumerate().any(|(other, candidate)| {
            other != index
                && candidates[index].len() < candidate.len()
                && candidates[index].is_subset(candidate)
        });
        if !is_subset_of_other {
            maximal.push(candidates[index].iter().copied().collect());
        }
    }
    maximal
}

fn build_max_weight_clique_forest(graph: &Graph, cliques: &[Vec<usize>]) -> CliqueForest {
    let sets: Vec<BTreeSet<usize>> = cliques
        .iter()
        .map(|clique| clique.iter().copied().collect())
        .collect();

    let mut weighted_edges: Vec<(usize, usize, usize)> = Vec::new();
    for a in 0..sets.len() {
        for b in (a + 1)..sets.len() {
            let weight = sets[a].intersection(&sets[b]).count();
            if weight > 0 {
                weighted_edges.push((a, b, weight));
            }
        }
    }

    weighted_edges.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });

    let mut parent: Vec<usize> = (0..sets.len()).collect();
    let mut rank = vec![0usize; sets.len()];
    let mut chosen = Vec::new();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    for (a, b, weight) in weighted_edges {
        let root_a = find(&mut parent, a);
        let root_b = find(&mut parent, b);
        if root_a == root_b {
            continue;
        }
        match rank[root_a].cmp(&rank[root_b]) {
            std::cmp::Ordering::Less => parent[root_a] = root_b,
            std::cmp::Ordering::Greater => parent[root_b] = root_a,
            std::cmp::Ordering::Equal => {
                parent[root_b] = root_a;
                rank[root_a] += 1;
            }
        }
        chosen.push((a, b, weight));
    }

    let nodes = sets
        .iter()
        .enumerate()
        .map(|(index, set)| {
            let vertex_ids: Vec<usize> = set.iter().copied().collect();
            CliqueNode {
                index,
                vertices: vertex_ids
                    .iter()
                    .map(|id| graph.name(*id).to_string())
                    .collect(),
                size: vertex_ids.len(),
                vertex_ids,
            }
        })
        .collect();

    let edges = chosen
        .into_iter()
        .map(|(from, to, weight)| {
            let separator_ids: Vec<usize> = sets[from].intersection(&sets[to]).copied().collect();
            CliqueForestEdge {
                from,
                to,
                separator: separator_ids
                    .iter()
                    .map(|id| graph.name(*id).to_string())
                    .collect(),
                weight,
                separator_ids,
            }
        })
        .collect();

    let mut roots: BTreeSet<usize> = (0..sets.len()).map(|id| find(&mut parent, id)).collect();
    CliqueForest {
        nodes,
        edges,
        component_count: roots.len(),
    }
}

pub fn map_named_order(
    graph: &Graph,
    order_names: &[String],
) -> Result<Vec<usize>, crate::errors::InputError> {
    if order_names.len() != graph.vertex_count() {
        return Err(crate::errors::InputError::InvalidOrderLength {
            expected: graph.vertex_count(),
            actual: order_names.len(),
        });
    }

    let mut used = vec![false; graph.vertex_count()];
    let mut ids = Vec::with_capacity(order_names.len());
    for name in order_names {
        let Some(id) = graph.id(name) else {
            return Err(crate::errors::InputError::UnknownOrderVertex(name.clone()));
        };
        if used[id] {
            return Err(crate::errors::InputError::DuplicateOrderVertex(
                name.clone(),
            ));
        }
        used[id] = true;
        ids.push(id);
    }
    Ok(ids)
}
