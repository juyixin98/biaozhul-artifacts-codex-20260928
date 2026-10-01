#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use chordal_api::graph::Graph;
use chordal_api::logging::RunLogger;

pub fn graph(vertices: &[&str], edges: &[(&str, &str)]) -> Graph {
    let vertices = vertices.iter().map(|value| value.to_string()).collect();
    let edges = edges
        .iter()
        .map(|(left, right)| (left.to_string(), right.to_string()))
        .collect();
    Graph::from_parts(Some(vertices), edges).expect("test graph must build")
}

pub fn implicit_graph(edges: &[(&str, &str)]) -> Graph {
    let edges = edges
        .iter()
        .map(|(left, right)| (left.to_string(), right.to_string()))
        .collect();
    Graph::from_parts(None, edges).expect("test graph must build")
}

pub fn labels(graph: &Graph, ids: &[usize]) -> Vec<String> {
    ids.iter().map(|id| graph.label(*id).to_owned()).collect()
}

pub fn reference_has_induced_cycle(graph: &Graph) -> bool {
    let n = graph.vertex_count();
    for start in 0..n {
        let mut path = vec![start];
        let mut used = vec![false; n];
        used[start] = true;
        if reference_dfs_cycle(graph, start, start, 0, &mut path, &mut used) {
            return true;
        }
    }
    false
}

fn reference_dfs_cycle(
    graph: &Graph,
    start: usize,
    current: usize,
    depth: usize,
    path: &mut Vec<usize>,
    used: &mut [bool],
) -> bool {
    for &next in graph.neighbors(current) {
        if next == start && depth >= 3 {
            let length = path.len();
            let mut chord = false;
            'outer: for i in 0..length {
                for j in (i + 2)..length {
                    if i == 0 && j == length - 1 {
                        continue;
                    }
                    if graph.has_edge(path[i], path[j]) {
                        chord = true;
                        break 'outer;
                    }
                }
            }
            if !chord {
                return true;
            }
        }
        if used[next] || next <= start {
            continue;
        }
        used[next] = true;
        path.push(next);
        if reference_dfs_cycle(graph, start, next, depth + 1, path, used) {
            return true;
        }
        path.pop();
        used[next] = false;
    }
    false
}

pub fn reference_maximal_cliques(graph: &Graph) -> Vec<BTreeSet<usize>> {
    let n = graph.vertex_count();
    let mut cliques = Vec::new();
    for mask in 0u128..(1u128 << n) {
        let members: BTreeSet<usize> = (0..n)
            .filter(|vertex| mask & (1u128 << vertex) != 0)
            .collect();
        if members.is_empty() {
            continue;
        }
        let is_clique = members
            .iter()
            .all(|u| members.iter().all(|v| u == v || graph.has_edge(*u, *v)));
        let is_maximal = is_clique
            && (0..n)
                .filter(|vertex| !members.contains(vertex))
                .all(|vertex| {
                    !members
                        .iter()
                        .all(|neighbor| graph.has_edge(vertex, *neighbor))
                });
        if is_maximal {
            cliques.push(members);
        }
    }
    cliques
}

pub fn reference_check_running_intersection(
    graph: &Graph,
    cliques: &[BTreeSet<usize>],
    edges: &[(usize, usize)],
) -> bool {
    let n = graph.vertex_count();
    let mut adjacency: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for vertex in 0..cliques.len() {
        adjacency.entry(vertex).or_default();
    }
    for (u, v) in edges {
        adjacency.entry(*u).or_default().push(*v);
        adjacency.entry(*v).or_default().push(*u);
    }

    for vertex in 0..n {
        let containing: Vec<usize> = cliques
            .iter()
            .enumerate()
            .filter_map(|(id, clique)| clique.contains(&vertex).then_some(id))
            .collect();
        if containing.is_empty() {
            return false;
        }
        if containing.len() == 1 {
            continue;
        }
        let root = containing[0];
        let targets: BTreeSet<usize> = containing.iter().copied().collect();
        let mut visited = BTreeSet::new();
        let mut stack = vec![root];
        visited.insert(root);
        while let Some(node) = stack.pop() {
            for neighbor in adjacency.get(&node).cloned().unwrap_or_default() {
                if visited.insert(neighbor) && cliques[neighbor].contains(&vertex) {
                    stack.push(neighbor);
                }
            }
        }
        if !targets.iter().all(|target| visited.contains(target)) {
            return false;
        }
    }
    true
}

pub fn test_logger(test_name: &str) -> RunLogger {
    RunLogger::new(format!("test-logs/rust/{test_name}.jsonl"))
}
