//! K-truss decomposition kernel.

use crate::graph::{Node, SimpleGraph};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PeelingOrder {
    #[default]
    Declaration,
    ReverseDeclaration,
    EndpointLex,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeDecomposition {
    pub edge_id: String,
    pub u: String,
    pub v: String,
    pub initial_support: usize,
    pub level: usize,
    pub witness_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Witness {
    pub witness_id: String,
    pub level: usize,
    pub node_ids: Vec<String>,
    pub edge_ids: Vec<String>,
    pub edge_count: usize,
    pub triangle_count: usize,
    pub minimum_support: usize,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct KernelCounters {
    pub triangle_count: usize,
    pub affected_triangle_updates: usize,
    pub support_decrements: usize,
    pub removed_edges: usize,
    pub peeling_rounds: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Decomposition {
    pub levels: Vec<usize>,
    pub initial_support: Vec<usize>,
    pub edges: Vec<EdgeDecomposition>,
    pub witnesses: Vec<Witness>,
    pub counters: KernelCounters,
}

#[derive(Debug, Clone)]
pub struct KernelError {
    pub status: u16,
    pub code: String,
    pub message: String,
}

impl Decomposition {
    pub fn edge_level(&self, edge_id: &str) -> Option<usize> {
        self.edges
            .iter()
            .find(|edge| edge.edge_id == edge_id)
            .map(|edge| edge.level)
    }
}

pub fn triangle_count(graph: &SimpleGraph) -> usize {
    graph.triangle_count()
}

pub fn edge_support(graph: &SimpleGraph, edge: usize) -> usize {
    graph.edge_support(edge)
}

pub fn decompose(graph: &SimpleGraph, order: PeelingOrder) -> Result<Decomposition, KernelError> {
    let edge_count = graph.edge_count();
    let node_count = graph.node_count();
    let mut support = Vec::with_capacity(edge_count);
    for edge in 0..edge_count {
        support.push(graph.edge_support(edge));
    }

    let triangle_count = support.iter().sum::<usize>() / 3;
    let mut active = vec![true; edge_count];
    let mut levels = vec![0usize; edge_count];
    let mut counters = KernelCounters {
        triangle_count,
        ..Default::default()
    };

    let mut queue = BinaryHeap::new();
    for edge in 0..edge_count {
        queue.push(QueueItem::new(graph, order, support[edge], edge));
    }

    // The level-k round removes the closure of edges with support below k-2.
    let max_level = node_count.max(2);
    for level in 3..=max_level {
        let removal_threshold = level.saturating_sub(3);
        let mut removed_in_round = 0usize;

        loop {
            let next = match queue.peek() {
                Some(item) if item.support <= removal_threshold => *item,
                _ => break,
            };
            queue.pop();

            let edge = next.edge;
            if !active[edge] || support[edge] != next.support {
                continue;
            }

            let (u, v) = graph.edge_endpoints(edge);
            active[edge] = false;
            levels[edge] = level - 1;
            removed_in_round += 1;
            counters.removed_edges += 1;

            for w in active_common_neighbors(graph, &active, u, v) {
                counters.affected_triangle_updates += 1;

                if let Some(uw) = active_edge_between(graph, &active, u, w) {
                    support[uw] -= 1;
                    counters.support_decrements += 1;
                    queue.push(QueueItem::new(graph, order, support[uw], uw));
                }
                if let Some(vw) = active_edge_between(graph, &active, v, w) {
                    support[vw] -= 1;
                    counters.support_decrements += 1;
                    queue.push(QueueItem::new(graph, order, support[vw], vw));
                }
            }
        }

        if removed_in_round > 0 {
            counters.peeling_rounds += 1;
        }
    }

    // Any surviving edge belongs to the densest level representable by this graph.
    for edge in 0..edge_count {
        if active[edge] {
            levels[edge] = max_level;
            counters.removed_edges += 0;
        }
    }

    // Empty graphs still expose the conventional base truss level as 2.
    let highest_level = levels.iter().copied().max().unwrap_or(2).max(2);
    let (witnesses, witness_by_edge) = build_witnesses(graph, &levels, highest_level);

    // Initial support is a static property of the normalized simple graph. Rebuilding it after
    // mutation keeps the response independent from bookkeeping inside the peeling loop.
    let initial_support = (0..edge_count)
        .map(|edge| graph.edge_support(edge))
        .collect::<Vec<_>>();

    let edges = graph
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| EdgeDecomposition {
            edge_id: edge.id.clone(),
            u: graph.node_ids[edge.u].clone(),
            v: graph.node_ids[edge.v].clone(),
            initial_support: initial_support[index],
            level: levels[index],
            witness_id: witness_by_edge
                .get(&index)
                .cloned()
                .unwrap_or_else(|| format!("k{}-missing", levels[index])),
        })
        .collect();

    Ok(Decomposition {
        levels,
        initial_support,
        edges,
        witnesses,
        counters,
    })
}

#[derive(Debug, Clone, Copy)]
struct QueueItem {
    support: usize,
    edge: usize,
    order_key: OrderKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OrderKey {
    declaration: usize,
    first: Reverse<Node>,
    second: Reverse<Node>,
}

impl QueueItem {
    fn new(graph: &SimpleGraph, order: PeelingOrder, support: usize, edge: usize) -> Self {
        let stored = graph.edge_endpoints(edge);
        let (mut first, mut second) = (stored.0, stored.1);
        if first > second {
            std::mem::swap(&mut first, &mut second);
        }
        let declaration = match order {
            PeelingOrder::Declaration => edge_count_minus_index(graph, edge),
            PeelingOrder::ReverseDeclaration => edge,
            PeelingOrder::EndpointLex => edge,
        };
        Self {
            support,
            edge,
            order_key: OrderKey {
                declaration,
                first: Reverse(first),
                second: Reverse(second),
            },
        }
    }
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap, so reverse support makes the smallest support pop first.
        Reverse(self.support)
            .cmp(&Reverse(other.support))
            .then(self.order_key.cmp(&other.order_key))
    }
}

fn edge_count_minus_index(graph: &SimpleGraph, edge: usize) -> usize {
    graph.edge_count().saturating_sub(edge + 1)
}

fn active_common_neighbors(graph: &SimpleGraph, active: &[bool], u: Node, v: Node) -> Vec<Node> {
    let u_neighbors: HashMap<Node, usize> = graph
        .neighbors(u)
        .iter()
        .filter(|item| active[item.edge])
        .map(|item| (item.node, item.edge))
        .collect();

    graph
        .neighbors(v)
        .iter()
        .filter(|item| active[item.edge])
        .filter_map(|item| u_neighbors.get(&item.node).map(|_| item.node))
        .collect()
}

fn active_edge_between(graph: &SimpleGraph, active: &[bool], u: Node, v: Node) -> Option<usize> {
    graph.neighbors(u).iter().find_map(|item| {
        if item.node == v && active[item.edge] {
            Some(item.edge)
        } else {
            None
        }
    })
}

fn build_witnesses(
    graph: &SimpleGraph,
    levels: &[usize],
    highest_level: usize,
) -> (Vec<Witness>, HashMap<usize, String>) {
    let mut witnesses = Vec::new();
    let mut witness_by_edge = HashMap::new();

    for level in 2..=highest_level {
        let edge_in_layer = |edge: usize| levels[edge] >= level;
        let components = connected_components(graph, edge_in_layer);

        for (component_index, component) in components.into_iter().enumerate() {
            let exact_edges: Vec<usize> = component
                .edges
                .iter()
                .copied()
                .filter(|&edge| levels[edge] == level)
                .collect();
            if exact_edges.is_empty() {
                continue;
            }

            let edge_set: HashSet<usize> = component.edges.iter().copied().collect();
            let mut supports = Vec::new();
            for &edge in &component.edges {
                let (u, v) = graph.edge_endpoints(edge);
                let support = graph
                    .common_neighbors(u, v)
                    .iter()
                    .filter(|&&w| {
                        edge_set.contains(&edge_between(graph, u, w).unwrap())
                            && edge_set.contains(&edge_between(graph, v, w).unwrap())
                    })
                    .count();
                supports.push(support);
            }
            let minimum_support = supports.into_iter().min().unwrap_or(0);
            let triangle_count = component
                .edges
                .iter()
                .map(|&edge| {
                    let (u, v) = graph.edge_endpoints(edge);
                    graph
                        .common_neighbors(u, v)
                        .iter()
                        .filter(|&&w| {
                            edge_set.contains(&edge_between(graph, u, w).unwrap())
                                && edge_set.contains(&edge_between(graph, v, w).unwrap())
                        })
                        .count()
                })
                .sum::<usize>()
                / 3;

            let mut node_ids: Vec<String> = component
                .nodes
                .iter()
                .map(|node| graph.node_ids[*node].clone())
                .collect();
            node_ids.sort();
            let mut edge_ids: Vec<String> = component
                .edges
                .iter()
                .map(|edge| graph.edges[*edge].id.clone())
                .collect();
            edge_ids.sort();

            witnesses.push(Witness {
                witness_id: format!("k{level}-c{component_index}"),
                level,
                node_ids,
                edge_ids,
                edge_count: component.edges.len(),
                triangle_count,
                minimum_support,
            });
            for edge in exact_edges {
                witness_by_edge.insert(edge, format!("k{level}-c{component_index}"));
            }
        }
    }

    (witnesses, witness_by_edge)
}

#[derive(Debug, Default)]
struct Component {
    nodes: HashSet<Node>,
    edges: HashSet<usize>,
}

fn connected_components(
    graph: &SimpleGraph,
    edge_filter: impl Fn(usize) -> bool,
) -> Vec<Component> {
    let mut seen_nodes = HashSet::new();
    let mut components = Vec::new();

    for start_edge in 0..graph.edge_count() {
        if !edge_filter(start_edge) {
            continue;
        }
        let start = graph.edges[start_edge].u;
        if seen_nodes.contains(&start) {
            continue;
        }

        let mut stack = vec![start];
        let mut component = Component::default();
        seen_nodes.insert(start);

        while let Some(node) = stack.pop() {
            component.nodes.insert(node);
            for neighbor in graph.neighbors(node) {
                if !edge_filter(neighbor.edge) || !component.edges.insert(neighbor.edge) {
                    continue;
                }
                if seen_nodes.insert(neighbor.node) {
                    stack.push(neighbor.node);
                }
                component.nodes.insert(neighbor.node);
            }
        }

        components.push(component);
    }

    components.sort_by_key(|component| {
        component
            .edges
            .iter()
            .map(|edge| graph.edges[*edge].id.clone())
            .min()
            .unwrap_or_default()
    });
    components
}

fn edge_between(graph: &SimpleGraph, u: Node, v: Node) -> Option<usize> {
    graph
        .neighbors(u)
        .iter()
        .find(|item| item.node == v)
        .map(|item| item.edge)
}
