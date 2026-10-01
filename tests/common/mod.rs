#![allow(dead_code)]

use std::collections::{BTreeSet, VecDeque};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct RefGraph {
    pub n: usize,
    pub adjacency: Vec<BTreeSet<usize>>,
}

impl RefGraph {
    pub fn new(n: usize, pairs: &[(usize, usize)]) -> Self {
        let mut adjacency = vec![BTreeSet::new(); n];
        for &(u, v) in pairs {
            assert_ne!(u, v);
            adjacency[u].insert(v);
            adjacency[v].insert(u);
        }
        Self { n, adjacency }
    }

    pub fn adjacent(&self, u: usize, v: usize) -> bool {
        self.adjacency[u].contains(&v)
    }

    pub fn chordless_cycle(&self) -> Option<Vec<usize>> {
        for size in 4..=self.n {
            for subset in combinations(0..self.n, size) {
                let mut permutation = subset.clone();
                if find_cycle_permutation(self, &mut permutation, 0) {
                    return Some(permutation);
                }
            }
        }
        None
    }

    pub fn chordal(&self) -> bool {
        self.chordless_cycle().is_none()
    }

    pub fn maximal_cliques(&self) -> Vec<Vec<usize>> {
        let mut found: Vec<Vec<usize>> = Vec::new();
        for size in (1..=self.n).rev() {
            for subset in combinations(0..self.n, size) {
                if !is_clique(self, &subset) {
                    continue;
                }
                let current: BTreeSet<usize> = subset.iter().copied().collect();
                let contained = found
                    .iter()
                    .any(|larger| current.is_subset(&larger.iter().copied().collect()));
                if !contained {
                    found.push(subset);
                }
            }
        }
        found.sort();
        found
    }
}

fn combinations(iter: std::ops::Range<usize>, size: usize) -> Vec<Vec<usize>> {
    let items: Vec<usize> = iter.collect();
    let mut output = Vec::new();
    let mut current = Vec::new();
    combine(&items, size, 0, &mut current, &mut output);
    output
}

fn combine(
    items: &[usize],
    size: usize,
    start: usize,
    current: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if current.len() == size {
        output.push(current.clone());
        return;
    }
    for index in start..items.len() {
        current.push(items[index]);
        combine(items, size, index + 1, current, output);
        current.pop();
    }
}

fn find_cycle_permutation(graph: &RefGraph, order: &mut [usize], start: usize) -> bool {
    if start == order.len() {
        return is_induced_cycle(graph, order);
    }
    for index in start..order.len() {
        order.swap(start, index);
        if find_cycle_permutation(graph, order, start + 1) {
            return true;
        }
        order.swap(start, index);
    }
    false
}

pub fn is_induced_cycle(graph: &RefGraph, cycle: &[usize]) -> bool {
    let len = cycle.len();
    if len < 4 {
        return false;
    }
    for index in 0..len {
        if !graph.adjacent(cycle[index], cycle[(index + 1) % len]) {
            return false;
        }
    }
    for i in 0..len {
        for j in (i + 2)..len {
            if i == 0 && j == len - 1 {
                continue;
            }
            if graph.adjacent(cycle[i], cycle[j]) {
                return false;
            }
        }
    }
    true
}

fn is_clique(graph: &RefGraph, vertices: &[usize]) -> bool {
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            if !graph.adjacent(vertices[i], vertices[j]) {
                return false;
            }
        }
    }
    true
}

pub fn post_json(port: u16, path: &str, body: Value) -> (u16, Value) {
    let payload = body.to_string();
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    stream.write_all(request.as_bytes()).unwrap();
    read_http_response(&mut stream)
}

fn read_http_response(stream: &mut TcpStream) -> (u16, Value) {
    let mut raw = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(size) => raw.extend_from_slice(&buffer[..size]),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(error) => panic!("HTTP read failed: {error}"),
        }
    }

    let text = String::from_utf8(raw).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap();
    let status: u16 = head
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let json = serde_json::from_str(body.trim()).unwrap_or(Value::Null);
    (status, json)
}

pub fn assert_forest_rip(json: &Value, vertices: &[&str], edges: &[(&str, &str)]) {
    let graph = RefGraph::new(
        vertices.len(),
        &edges
            .iter()
            .map(|(u, v)| {
                (
                    vertices.iter().position(|name| name == u).unwrap(),
                    vertices.iter().position(|name| name == v).unwrap(),
                )
            })
            .collect::<Vec<_>>(),
    );

    let cliques_json = &json["maximal_cliques"];
    let cliques: Vec<BTreeSet<usize>> = cliques_json
        .as_array()
        .unwrap()
        .iter()
        .map(|clique| {
            clique
                .as_array()
                .unwrap()
                .iter()
                .map(|vertex| {
                    let name = vertex.as_str().unwrap();
                    vertices.iter().position(|item| *item == name).unwrap()
                })
                .collect()
        })
        .collect();

    for (u, v) in &graph.adjacency.iter().enumerate().flat_map(|(u, neighbors)| {
        neighbors
            .iter()
            .filter(move |v| **v > u)
            .map(move |v| (u, *v))
            .collect::<Vec<_>>()
    }) {
        assert!(
            cliques.iter().any(|clique| clique.contains(&u) && clique.contains(&v)),
            "edge {u}-{v} is not covered by returned cliques"
        );
    }

    let forest_edges: Vec<(usize, usize)> = json["clique_forest"]["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            (
                edge["from"].as_u64().unwrap() as usize,
                edge["to"].as_u64().unwrap() as usize,
            )
        })
        .collect();

    let mut adjacency = vec![BTreeSet::new(); cliques.len()];
    for (u, v) in forest_edges {
        adjacency[u].insert(v);
        adjacency[v].insert(u);
    }

    for vertex in 0..graph.n {
        let nodes: Vec<usize> = cliques
            .iter()
            .enumerate()
            .filter_map(|(id, clique)| clique.contains(&vertex).then_some(id))
            .collect();
        assert!(!nodes.is_empty(), "vertex {vertex} uncovered");
        let target: BTreeSet<usize> = nodes.iter().copied().collect();
        let mut seen = BTreeSet::from([nodes[0]]);
        let mut queue = VecDeque::from([nodes[0]]);
        while let Some(node) = queue.pop_front() {
            for next in &adjacency[node] {
                if target.contains(next) && seen.insert(*next) {
                    queue.push_back(*next);
                }
            }
        }
        assert_eq!(seen, target, "RIP failed for vertex {vertex}");
    }
}
