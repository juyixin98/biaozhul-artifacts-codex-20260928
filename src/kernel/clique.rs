use std::collections::{HashMap, HashSet};

use crate::graph::Graph;
use crate::witness::WitnessCheck;

#[derive(Debug, Clone)]
pub struct Clique {
    pub id: usize,
    pub source_position: Option<usize>,
    pub source_vertex: Option<usize>,
    pub vertices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CliqueTreeEdge {
    pub parent: usize,
    pub child: usize,
    pub separator: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CliqueTree {
    pub root: usize,
    pub cliques: Vec<Clique>,
    pub edges: Vec<CliqueTreeEdge>,
}

#[derive(Debug, Clone)]
pub struct CliqueTreeCheck {
    pub valid: bool,
    pub reason: String,
    pub clique_cover_check: WitnessCheck,
    pub edge_cover_check: WitnessCheck,
    pub running_intersection_check: WitnessCheck,
    pub separator_check: WitnessCheck,
}

pub fn extract_maximal_cliques(graph: &Graph, order: &[usize]) -> Vec<Clique> {
    let n = graph.vertex_count();
    let mut position = vec![usize::MAX; n];
    for (index, vertex) in order.iter().enumerate() {
        position[*vertex] = index;
    }

    let mut candidates: Vec<Clique> = Vec::new();
    for (index, &vertex) in order.iter().enumerate() {
        let mut members = vec![vertex];
        let mut later_neighbors: Vec<usize> = graph
            .neighbors(vertex)
            .iter()
            .copied()
            .filter(|neighbor| position[*neighbor] > index)
            .collect();
        later_neighbors.sort_unstable();
        members.extend(later_neighbors);

        let contained = candidates.iter().any(|existing| {
            members
                .iter()
                .all(|vertex| existing.vertices.contains(vertex))
        });
        if !contained {
            candidates.push(Clique {
                id: candidates.len(),
                source_position: Some(index),
                source_vertex: Some(vertex),
                vertices: members,
            });
        }
    }

    if n == 0 {
        candidates.push(Clique {
            id: 0,
            source_position: None,
            source_vertex: None,
            vertices: Vec::new(),
        });
    }

    candidates.sort_by_key(|clique| clique.source_position);
    for (id, clique) in candidates.iter_mut().enumerate() {
        clique.id = id;
    }

    candidates
}

pub fn build_clique_tree(graph: &Graph, cliques: &[Clique], order: &[usize]) -> CliqueTree {
    let n = graph.vertex_count();
    if n == 0 {
        return CliqueTree {
            root: 0,
            cliques: cliques.to_vec(),
            edges: Vec::new(),
        };
    }

    let mut position = vec![usize::MAX; n];
    for (index, vertex) in order.iter().enumerate() {
        position[*vertex] = index;
    }

    let root = 0usize;
    let mut edges = Vec::new();
    for clique in cliques.iter() {
        let source_position = clique
            .source_position
            .expect("non-empty graph cliques have source positions");
        let later_set: HashSet<usize> = clique
            .vertices
            .iter()
            .copied()
            .filter(|vertex| position[*vertex] > source_position)
            .collect();

        let parent = if later_set.is_empty() {
            root
        } else {
            let source_index = cliques
                .iter()
                .position(|candidate| {
                    candidate
                        .source_position
                        .map_or(false, |candidate_position| {
                            candidate_position < source_position
                                && later_set
                                    .iter()
                                    .all(|vertex| candidate.vertices.contains(vertex))
                        })
                })
                .unwrap_or(root);
            source_index
        };

        if clique.id == root && parent == root {
            continue;
        }

        let parent_set: HashSet<usize> = cliques[parent].vertices.iter().copied().collect();
        let mut separator: Vec<usize> = clique
            .vertices
            .iter()
            .copied()
            .filter(|vertex| parent_set.contains(vertex))
            .collect();
        separator.sort_unstable();
        edges.push(CliqueTreeEdge {
            parent,
            child: clique.id,
            separator,
        });
    }

    CliqueTree {
        root,
        cliques: cliques.to_vec(),
        edges,
    }
}

pub fn check_clique_tree(graph: &Graph, tree: &CliqueTree) -> CliqueTreeCheck {
    let clique_cover_check = check_cliques_cover_vertices(graph, tree);
    let edge_cover_check = check_cliques_cover_edges(graph, tree);
    let separator_check = check_separators(graph, tree);
    let running_intersection_check = check_running_intersection(graph, tree);
    let tree_check = check_underlying_tree(graph, tree);

    let checks = [
        &clique_cover_check,
        &edge_cover_check,
        &separator_check,
        &running_intersection_check,
        &tree_check,
    ];
    let valid = checks.iter().all(|check| check.valid);
    let reason = if valid {
        "clique tree is connected and acyclic, covers vertices and edges, and satisfies the running intersection property".to_owned()
    } else {
        checks
            .iter()
            .find(|check| !check.valid)
            .map(|check| check.reason.clone())
            .unwrap_or_else(|| "clique tree validation failed".to_owned())
    };

    CliqueTreeCheck {
        valid,
        reason,
        clique_cover_check,
        edge_cover_check,
        running_intersection_check,
        separator_check,
    }
}

fn check_cliques_cover_vertices(graph: &Graph, tree: &CliqueTree) -> WitnessCheck {
    for vertex in 0..graph.vertex_count() {
        let covered = tree
            .cliques
            .iter()
            .any(|clique| clique.vertices.contains(&vertex));
        if !covered {
            return invalid(format!(
                "vertex {} is not covered by any clique",
                graph.label(vertex)
            ));
        }
    }

    for clique in &tree.cliques {
        for (i, &left) in clique.vertices.iter().enumerate() {
            for &right in clique.vertices.iter().skip(i + 1) {
                if !graph.has_edge(left, right) {
                    return invalid(format!(
                        "reported clique contains nonadjacent vertices {} and {}",
                        graph.label(left),
                        graph.label(right)
                    ));
                }
            }
        }
    }

    valid("every vertex is covered, and every reported set is a clique".to_owned())
}

fn check_cliques_cover_edges(graph: &Graph, tree: &CliqueTree) -> WitnessCheck {
    for u in 0..graph.vertex_count() {
        for &v in graph.neighbors(u) {
            if u < v {
                let covered = tree
                    .cliques
                    .iter()
                    .any(|clique| clique.vertices.contains(&u) && clique.vertices.contains(&v));
                if !covered {
                    return invalid(format!(
                        "edge ({}, {}) is not covered by one maximal clique",
                        graph.label(u),
                        graph.label(v)
                    ));
                }
            }
        }
    }

    let mut membership: HashMap<usize, Vec<usize>> = HashMap::new();
    for clique in &tree.cliques {
        for &vertex in &clique.vertices {
            membership.entry(vertex).or_default().push(clique.id);
        }
    }
    for edge in &tree.edges {
        for &vertex in &edge.separator {
            if !tree.cliques[edge.parent].vertices.contains(&vertex)
                || !tree.cliques[edge.child].vertices.contains(&vertex)
            {
                return invalid(format!(
                    "separator on edge {}->{} contains a vertex absent from an endpoint",
                    edge.parent, edge.child
                ));
            }
        }
    }

    let sets: Vec<HashSet<usize>> = tree
        .cliques
        .iter()
        .map(|clique| clique.vertices.iter().copied().collect())
        .collect();
    for (i, current) in sets.iter().enumerate() {
        for (j, other) in sets.iter().enumerate() {
            if i != j && current.is_subset(other) {
                return invalid(format!(
                    "clique {i} is contained in clique {j} and is not maximal"
                ));
            }
        }
    }

    let _ = membership;
    valid(
        "every edge is covered, all cliques are maximal, and separators are endpoint intersections"
            .to_owned(),
    )
}

fn check_separators(graph: &Graph, tree: &CliqueTree) -> WitnessCheck {
    for edge in &tree.edges {
        let left: HashSet<usize> = tree.cliques[edge.parent].vertices.iter().copied().collect();
        let right: HashSet<usize> = tree.cliques[edge.child].vertices.iter().copied().collect();
        let mut intersection: Vec<usize> = left.intersection(&right).copied().collect();
        intersection.sort_unstable();
        if intersection != edge.separator {
            return invalid(format!(
                "edge {}->{} separator does not equal its clique intersection",
                edge.parent, edge.child
            ));
        }
    }
    valid("every tree edge separator equals the intersection of its endpoint cliques".to_owned())
}

fn check_underlying_tree(_graph: &Graph, tree: &CliqueTree) -> WitnessCheck {
    if tree.cliques.is_empty() {
        return invalid("clique tree has no cliques".to_owned());
    }
    if tree.root >= tree.cliques.len() {
        return invalid(format!("clique tree root {} is out of range", tree.root));
    }
    if tree.edges.len() != tree.cliques.len().saturating_sub(1) {
        return invalid(format!(
            "clique forest has {} edges for {} cliques; a single tree requires {}",
            tree.edges.len(),
            tree.cliques.len(),
            tree.cliques.len().saturating_sub(1)
        ));
    }

    let mut adjacency = vec![Vec::new(); tree.cliques.len()];
    for edge in &tree.edges {
        adjacency[edge.parent].push(edge.child);
        adjacency[edge.child].push(edge.parent);
    }

    let mut visited = vec![false; tree.cliques.len()];
    let mut stack = vec![(tree.root, usize::MAX)];
    while let Some((node, parent)) = stack.pop() {
        if visited[node] {
            return invalid("clique tree contains a cycle".to_owned());
        }
        visited[node] = true;
        for &neighbor in &adjacency[node] {
            if neighbor != parent {
                stack.push((neighbor, node));
            }
        }
    }

    if visited.iter().all(|value| *value) {
        valid("maximal-clique nodes form one connected tree".to_owned())
    } else {
        invalid("clique tree is disconnected".to_owned())
    }
}

fn check_running_intersection(graph: &Graph, tree: &CliqueTree) -> WitnessCheck {
    let mut adjacency = vec![Vec::new(); tree.cliques.len()];
    for edge in &tree.edges {
        adjacency[edge.parent].push(edge.child);
        adjacency[edge.child].push(edge.parent);
    }

    for vertex in 0..graph.vertex_count() {
        let containing: Vec<usize> = tree
            .cliques
            .iter()
            .filter(|clique| clique.vertices.contains(&vertex))
            .map(|clique| clique.id)
            .collect();
        if containing.len() <= 1 {
            continue;
        }

        let root = containing[0];
        let targets: HashSet<usize> = containing.iter().copied().collect();
        let mut visited = vec![false; tree.cliques.len()];
        let mut stack = vec![root];
        visited[root] = true;
        while let Some(node) = stack.pop() {
            for &neighbor in &adjacency[node] {
                if visited[neighbor] || !tree.cliques[neighbor].vertices.contains(&vertex) {
                    continue;
                }
                visited[neighbor] = true;
                stack.push(neighbor);
            }
        }

        for clique_id in targets {
            if !visited[clique_id] {
                return invalid(format!(
                    "cliques containing vertex {} do not induce a connected subtree",
                    graph.label(vertex)
                ));
            }
        }
    }

    valid("for each vertex, all containing cliques induce a connected subtree".to_owned())
}

fn invalid(reason: String) -> WitnessCheck {
    WitnessCheck {
        valid: false,
        reason,
    }
}

fn valid(reason: String) -> WitnessCheck {
    WitnessCheck {
        valid: true,
        reason,
    }
}
