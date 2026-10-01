use std::collections::{BTreeSet, VecDeque};

use serde::Serialize;
use serde_json::{json, Value};

use crate::graph::Graph;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChordlessCycle {
    pub length: usize,
    pub vertex_ids: Vec<usize>,
    pub vertices: Vec<String>,
    pub edges: Vec<[String; 2]>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PeoViolation {
    pub vertex_id: usize,
    pub vertex: String,
    pub later_neighbor_ids: Vec<usize>,
    pub later_neighbors: Vec<String>,
    pub missing_edge: [String; 2],
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Default, PartialEq, Eq)]
pub struct CheckReport {
    pub passed: bool,
    pub failures: Vec<String>,
    pub details: Value,
}

impl CheckReport {
    fn pass(details: Value) -> Self {
        Self {
            passed: true,
            failures: Vec::new(),
            details,
        }
    }

    fn fail(failures: Vec<String>, details: Value) -> Self {
        Self {
            passed: false,
            failures,
            details,
        }
    }
}

pub fn peo_violation(graph: &Graph, order: &[usize]) -> Option<PeoViolation> {
    let n = graph.vertex_count();
    if order.len() != n {
        return None;
    }
    let mut position = vec![0usize; n];
    for (position_value, vertex) in order.iter().enumerate() {
        position[*vertex] = position_value;
    }

    for vertex in order {
        let mut later_neighbors: Vec<usize> = graph
            .neighbors(*vertex)
            .iter()
            .copied()
            .filter(|neighbor| position[*neighbor] > position[*vertex])
            .collect();
        later_neighbors.sort_unstable();

        for i in 0..later_neighbors.len() {
            for j in (i + 1)..later_neighbors.len() {
                let a = later_neighbors[i];
                let b = later_neighbors[j];
                if !graph.adjacent(a, b) {
                    return Some(PeoViolation {
                        vertex_id: *vertex,
                        vertex: graph.name(*vertex).to_string(),
                        later_neighbor_ids: later_neighbors.clone(),
                        later_neighbors: later_neighbors
                            .iter()
                            .map(|id| graph.name(*id).to_string())
                            .collect(),
                        missing_edge: [graph.name(a).to_string(), graph.name(b).to_string()],
                        reason: "two later neighbors are not adjacent, so they cannot be a clique"
                            .to_string(),
                    });
                }
            }
        }
    }
    None
}

pub fn find_induced_cycle(graph: &Graph) -> Option<ChordlessCycle> {
    for root in 0..graph.vertex_count() {
        let terminals: Vec<usize> = graph.neighbors(root).iter().copied().collect();
        for pair_start in 0..terminals.len() {
            for pair_end in (pair_start + 1)..terminals.len() {
                let start = terminals[pair_start];
                let target = terminals[pair_end];
                if graph.adjacent(start, target) {
                    continue;
                }

                let Some(path) =
                    shortest_path_without_other_root_neighbors(graph, root, start, target)
                else {
                    continue;
                };

                let mut cycle = Vec::with_capacity(path.len() + 1);
                cycle.push(root);
                cycle.extend(path);
                if let Ok(reason) = validate_chordless_cycle(graph, &cycle) {
                    return Some(cycle_witness(graph, cycle, reason));
                }
            }
        }
    }
    None
}

fn shortest_path_without_other_root_neighbors(
    graph: &Graph,
    root: usize,
    start: usize,
    target: usize,
) -> Option<Vec<usize>> {
    let n = graph.vertex_count();
    let mut previous = vec![usize::MAX; n];
    let mut queue = VecDeque::from([start]);
    previous[start] = start;

    while let Some(current) = queue.pop_front() {
        if current == target {
            let mut path = vec![target];
            let mut cursor = target;
            while cursor != start {
                cursor = previous[cursor];
                path.push(cursor);
            }
            path.reverse();
            return Some(path);
        }

        for next in graph.neighbors(current) {
            if previous[*next] != usize::MAX {
                continue;
            }
            let allowed = *next == target || (*next != root && !graph.adjacent(root, *next));
            if allowed {
                previous[*next] = current;
                queue.push_back(*next);
            }
        }
    }
    None
}

pub fn validate_chordless_cycle(graph: &Graph, cycle: &[usize]) -> Result<String, String> {
    let len = cycle.len();
    if len < 4 {
        return Err(format!("cycle length {len} is shorter than 4"));
    }

    let unique: BTreeSet<usize> = cycle.iter().copied().collect();
    if unique.len() != len {
        return Err("cycle contains a repeated vertex".to_string());
    }

    for index in 0..len {
        let a = cycle[index];
        let b = cycle[(index + 1) % len];
        if !graph.adjacent(a, b) {
            return Err(format!(
                "missing cycle edge between positions {index} and {}",
                index + 1
            ));
        }
    }

    for i in 0..len {
        for j in (i + 2)..len {
            if i == 0 && j == len - 1 {
                continue;
            }
            if graph.adjacent(cycle[i], cycle[j]) {
                return Err(format!("chord found between positions {i} and {j}"));
            }
        }
    }

    Ok(format!(
        "independently checked: {len} cycle edges exist and no non-consecutive edge exists"
    ))
}

fn cycle_witness(graph: &Graph, cycle: Vec<usize>, reason: String) -> ChordlessCycle {
    let edges = cycle
        .iter()
        .enumerate()
        .map(|(index, vertex)| {
            let next = cycle[(index + 1) % cycle.len()];
            [
                graph.name(*vertex).to_string(),
                graph.name(next).to_string(),
            ]
        })
        .collect();
    ChordlessCycle {
        length: cycle.len(),
        vertices: cycle.iter().map(|id| graph.name(*id).to_string()).collect(),
        vertex_ids: cycle,
        edges,
        reason,
    }
}

pub fn maximal_cliques_check(graph: &Graph, cliques: &[Vec<usize>]) -> CheckReport {
    let mut failures = Vec::new();
    let mut unique_cliques = BTreeSet::new();

    for (index, clique) in cliques.iter().enumerate() {
        let sorted: BTreeSet<usize> = clique.iter().copied().collect();
        if sorted.len() != clique.len() {
            failures.push(format!("clique {index} contains a repeated vertex"));
        }
        for vertex in clique {
            if *vertex >= graph.vertex_count() {
                failures.push(format!(
                    "clique {index} contains unknown vertex id {vertex}"
                ));
            }
        }
        for i in 0..clique.len() {
            for j in (i + 1)..clique.len() {
                if !graph.adjacent(clique[i], clique[j]) {
                    failures.push(format!(
                        "clique {index} is not a clique: {} - {} is missing",
                        graph.name(clique[i]),
                        graph.name(clique[j])
                    ));
                }
            }
        }
        if !unique_cliques.insert(sorted) {
            failures.push(format!("clique {index} is duplicated"));
        }
    }

    for (index, clique) in cliques.iter().enumerate() {
        let members: BTreeSet<usize> = clique.iter().copied().collect();
        for vertex in 0..graph.vertex_count() {
            if members.contains(&vertex) {
                continue;
            }
            if clique.iter().all(|member| graph.adjacent(vertex, *member)) {
                failures.push(format!(
                    "clique {index} is not maximal: vertex {} can be added",
                    graph.name(vertex)
                ));
            }
        }
    }

    let details = json!({
        "clique_count": cliques.len(),
        "clique_sizes": cliques.iter().map(|clique| clique.len()).collect::<Vec<_>>()
    });
    if failures.is_empty() {
        CheckReport::pass(details)
    } else {
        CheckReport::fail(failures, details)
    }
}

pub fn coverage_check(graph: &Graph, cliques: &[Vec<usize>]) -> CheckReport {
    let mut failures = Vec::new();
    let mut covered_vertices = BTreeSet::new();
    let mut covered_edges = BTreeSet::new();

    for clique in cliques {
        for vertex in clique {
            covered_vertices.insert(*vertex);
        }
        for i in 0..clique.len() {
            for j in (i + 1)..clique.len() {
                let (a, b) = if clique[i] < clique[j] {
                    (clique[i], clique[j])
                } else {
                    (clique[j], clique[i])
                };
                covered_edges.insert((a, b));
            }
        }
    }

    for vertex in 0..graph.vertex_count() {
        if !covered_vertices.contains(&vertex) {
            failures.push(format!("vertex {} is not covered", graph.name(vertex)));
        }
    }
    for u in 0..graph.vertex_count() {
        for v in graph.neighbors(u).range((u + 1)..) {
            if !covered_edges.contains(&(u, *v)) {
                failures.push(format!(
                    "edge {} - {} is not covered",
                    graph.name(u),
                    graph.name(*v)
                ));
            }
        }
    }

    let details = json!({
        "covered_vertices": covered_vertices.len(),
        "covered_edges": covered_edges.len(),
        "graph_edges": graph.edge_count()
    });
    if failures.is_empty() {
        CheckReport::pass(details)
    } else {
        CheckReport::fail(failures, details)
    }
}

pub fn clique_forest_check(
    graph: &Graph,
    cliques: &[Vec<usize>],
    edges: &[(usize, usize)],
) -> CheckReport {
    let mut failures = Vec::new();
    let clique_sets: Vec<BTreeSet<usize>> = cliques
        .iter()
        .map(|clique| clique.iter().copied().collect())
        .collect();

    let mut forest_adjacency = vec![BTreeSet::new(); cliques.len()];
    let mut canonical_edges = BTreeSet::new();
    for (a, b) in edges {
        if *a >= cliques.len() || *b >= cliques.len() {
            failures.push(format!("forest edge {a}-{b} references an unknown clique"));
            continue;
        }
        if a == b {
            failures.push(format!("forest edge {a}-{b} is a self loop"));
            continue;
        }
        let key = if a < b { (*a, *b) } else { (*b, *a) };
        if !canonical_edges.insert(key) {
            failures.push(format!("forest edge {a}-{b} is duplicated"));
        }
        forest_adjacency[key.0].insert(key.1);
        forest_adjacency[key.1].insert(key.0);
    }

    for component in graph.components() {
        let component_set: BTreeSet<usize> = component.iter().copied().collect();
        let mut clique_nodes = BTreeSet::new();

        for (clique_id, clique) in clique_sets.iter().enumerate() {
            if clique.is_empty() {
                continue;
            }
            let first = *clique.first().unwrap();
            if component_set.contains(&first) {
                if clique.iter().any(|vertex| !component_set.contains(vertex)) {
                    failures.push(format!("clique {clique_id} crosses connected components"));
                }
                clique_nodes.insert(clique_id);
            }
        }

        let internal_edges = canonical_edges
            .iter()
            .filter(|(a, b)| clique_nodes.contains(a) && clique_nodes.contains(b))
            .count();
        if !clique_nodes.is_empty() && internal_edges + 1 != clique_nodes.len() {
            failures.push(format!(
                "clique nodes for a graph component are not one tree: {} nodes, {internal_edges} edges",
                clique_nodes.len()
            ));
        }

        if let Some(first) = clique_nodes.first().copied() {
            let mut seen = BTreeSet::from([first]);
            let mut stack = vec![first];
            while let Some(node) = stack.pop() {
                for next in &forest_adjacency[node] {
                    if clique_nodes.contains(next) && seen.insert(*next) {
                        stack.push(*next);
                    }
                }
            }
            if seen != clique_nodes {
                failures.push("clique forest is disconnected inside a graph component".to_string());
            }
        }
    }

    for vertex in 0..graph.vertex_count() {
        let containing: Vec<usize> = clique_sets
            .iter()
            .enumerate()
            .filter_map(|(id, clique)| clique.contains(&vertex).then_some(id))
            .collect();
        if containing.is_empty() {
            failures.push(format!(
                "vertex {} appears in no clique",
                graph.name(vertex)
            ));
            continue;
        }

        let nodes: BTreeSet<usize> = containing.iter().copied().collect();
        let first = containing[0];
        let mut seen = BTreeSet::from([first]);
        let mut stack = vec![first];
        while let Some(node) = stack.pop() {
            for next in &forest_adjacency[node] {
                if nodes.contains(next) && seen.insert(*next) {
                    stack.push(*next);
                }
            }
        }
        if seen != nodes {
            failures.push(format!(
                "running intersection property fails for vertex {}",
                graph.name(vertex)
            ));
        }
    }

    let separator_sizes: Vec<usize> = canonical_edges
        .iter()
        .map(|(a, b)| clique_sets[*a].intersection(&clique_sets[*b]).count())
        .collect();
    let details = json!({
        "forest_edges": canonical_edges.len(),
        "separator_sizes": separator_sizes
    });
    if failures.is_empty() {
        CheckReport::pass(details)
    } else {
        CheckReport::fail(failures, details)
    }
}
