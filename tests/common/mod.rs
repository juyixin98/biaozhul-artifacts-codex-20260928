#![allow(dead_code)]

use dag_path_cover::{model::GraphInput, router, Config};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use tower::ServiceExt;

/// An independently enumerated chain partition. Each outer item is one ordered path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleAnswer {
    pub minimum_path_count: usize,
    pub minimum_partitions: Vec<Vec<Vec<usize>>>,
}

#[derive(Clone)]
struct PartitionContext {
    adjacency: Vec<BTreeSet<usize>>,
    blocks: Vec<Vec<usize>>,
}

/// Enumerates set partitions and all path orders without using the project algorithm.
/// This is intentionally exponential and is only used for small synthetic fixtures.
pub fn enumerate_minimum_path_partitions(value: &Value) -> OracleAnswer {
    let n = value["vertices"].as_u64().unwrap() as usize;
    let mut adjacency = vec![BTreeSet::new(); n];

    for edge in value["edges"].as_array().unwrap() {
        let from = edge["from"].as_u64().unwrap() as usize;
        let to = edge["to"].as_u64().unwrap() as usize;
        adjacency[from].insert(to);
    }

    let mut valid_partitions: BTreeSet<Vec<Vec<usize>>> = BTreeSet::new();
    let ctx = PartitionContext {
        adjacency,
        blocks: Vec::new(),
    };

    enumerate_partitions(0, n, ctx, &mut valid_partitions);

    if valid_partitions.is_empty() {
        panic!("independent oracle could not find the singleton partition");
    }

    let minimum_path_count = valid_partitions
        .iter()
        .map(Vec::len)
        .min()
        .expect("nonempty oracle result");
    let minimum_partitions = valid_partitions
        .into_iter()
        .filter(|partition| partition.len() == minimum_path_count)
        .collect();

    OracleAnswer {
        minimum_path_count,
        minimum_partitions,
    }
}

fn enumerate_partitions(
    vertex: usize,
    n: usize,
    ctx: PartitionContext,
    answers: &mut BTreeSet<Vec<Vec<usize>>>,
) {
    if vertex == n {
        for ordered in order_blocks(&ctx.blocks, &ctx.adjacency) {
            answers.insert(ordered);
        }
        return;
    }

    for block_index in 0..ctx.blocks.len() {
        let mut next = ctx.clone();
        next.blocks[block_index].push(vertex);
        enumerate_partitions(vertex + 1, n, next, answers);
    }

    let mut next = ctx.clone();
    next.blocks.push(vec![vertex]);
    enumerate_partitions(vertex + 1, n, next, answers);
}

fn order_blocks(blocks: &[Vec<usize>], adjacency: &[BTreeSet<usize>]) -> Vec<Vec<Vec<usize>>> {
    let mut completed: Vec<Vec<Vec<usize>>> = vec![Vec::new()];

    for block in blocks {
        let mut permutation = block.clone();
        let mut valid_orders = Vec::new();
        let mut permutations = Vec::new();
        permute(&mut permutation, 0, &mut permutations);

        for candidate in permutations {
            let valid = candidate
                .windows(2)
                .all(|pair| adjacency[pair[0]].contains(&pair[1]));
            if valid {
                valid_orders.push(candidate);
            }
        }

        let mut expanded = Vec::new();
        for existing in &completed {
            for order in &valid_orders {
                let mut next = existing.clone();
                next.push(order.clone());
                expanded.push(next);
            }
        }
        completed = expanded;
    }

    for partition in &mut completed {
        partition.sort_by_key(|path| *path.first().unwrap());
    }
    completed.sort();
    completed.dedup();
    completed
}

fn permute(items: &mut Vec<usize>, start: usize, answers: &mut Vec<Vec<usize>>) {
    if start == items.len() {
        answers.push(items.clone());
        return;
    }

    for index in start..items.len() {
        items.swap(start, index);
        permute(items, start + 1, answers);
        items.swap(start, index);
    }
}

pub async fn post_path_cover(input: &Value, run_id: &str) -> (http_status::StatusCode, Value) {
    let app = router(Config::default());

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/v1/path-cover/minimum")
                .header("content-type", "application/json")
                .header("x-run-id", run_id)
                .body(axum::body::Body::from(input.to_string()))
                .expect("request"),
        )
        .await
        .expect("response");

    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let value: Value = serde_json::from_slice(&bytes).expect("JSON response");
    (status, value)
}

pub async fn get_health() -> (http_status::StatusCode, Value) {
    let app = router(Config::default());

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("GET")
                .uri("/health")
                .body(axum::body::Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let value: Value = serde_json::from_slice(&bytes).expect("JSON response");
    (status, value)
}

pub fn input_from_sample(name: &str) -> (GraphInput, Value) {
    let path = format!("samples/{name}");
    let contents = std::fs::read_to_string(path).expect("sample file");
    let value: Value = serde_json::from_str(&contents).expect("sample JSON");
    let input: GraphInput = serde_json::from_value(value.clone()).expect("typed sample input");
    (input, value)
}

pub fn assert_concrete_partition(answer: &OracleAnswer, actual: &[Value]) {
    let actual_paths: Vec<Vec<usize>> = actual
        .iter()
        .map(|path| {
            path.as_array()
                .expect("path")
                .iter()
                .map(|vertex| vertex.as_u64().expect("vertex") as usize)
                .collect()
        })
        .collect();

    assert!(
        answer.minimum_partitions.contains(&actual_paths),
        "actual partition {actual_paths:?} is not one of the independently enumerated minimum partitions {:#?}",
        answer.minimum_partitions
    );
}

pub fn expected_error_json(code: &str, run_id: &str) -> Value {
    json!({
        "status": "error",
        "code": code,
        "run_id": run_id
    })
}

mod http_status {
    pub use axum::http::StatusCode;
}
