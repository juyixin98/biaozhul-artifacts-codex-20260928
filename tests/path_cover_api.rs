mod common;

use axum::http::StatusCode;
use common::{
    assert_concrete_partition, enumerate_minimum_path_partitions, expected_error_json, get_health,
    input_from_sample, post_path_cover,
};
use serde_json::{json, Value};

#[tokio::test]
async fn health_reports_service_version() {
    let (status, body) = get_health().await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], json!("ok"));
    assert_eq!(body["service"], json!("dag-path-cover"));
    assert_eq!(body["version"], json!(env!("CARGO_PKG_VERSION")));
}

#[tokio::test]
async fn zero_edge_graph_returns_every_single_vertex_path() {
    let run_id = "integration-zero-edges-0001";
    let (_typed, input) = input_from_sample("zero-edges.json");
    let oracle = enumerate_minimum_path_partitions(&input);

    let (status, body) = post_path_cover(&input, run_id).await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["run_id"], json!(run_id));
    assert_eq!(body["version"], json!(env!("CARGO_PKG_VERSION")));
    assert_eq!(body["graph"]["vertices"], json!(4));
    assert_eq!(body["graph"]["distinct_edges"], json!(0));
    assert_eq!(body["matching"]["size"], json!(0));
    assert_eq!(body["matching"]["edges"], json!([]));
    assert_eq!(body["path_count"], json!(4));
    assert_eq!(body["minimum_path_count"], json!(4));
    assert_eq!(body["paths"], json!([[0], [1], [2], [3]]));
    assert_eq!(oracle.minimum_path_count, 4);
    assert_concrete_partition(&oracle, body["paths"].as_array().unwrap());
    assert_eq!(body["witness"]["accepted"], json!(true));
    assert!(body["trace"]["steps"]
        .as_array()
        .unwrap()
        .iter()
        .any(|step| step
            .as_str()
            .unwrap()
            .contains("cycle check found no directed cycle")));
}

#[tokio::test]
async fn diamond_dag_reduces_to_two_paths() {
    let run_id = "integration-diamond-0001";
    let (_typed, input) = input_from_sample("diamond.json");
    let oracle = enumerate_minimum_path_partitions(&input);

    let (status, body) = post_path_cover(&input, run_id).await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(oracle.minimum_path_count, 2);
    assert_eq!(body["minimum_path_count"], json!(2));
    assert_eq!(body["matching"]["size"], json!(2));
    assert_eq!(
        body["path_count"].as_u64().unwrap() + body["matching"]["size"].as_u64().unwrap(),
        4
    );
    assert_eq!(body["paths"], json!([[0, 1, 3], [2]]));
    assert_concrete_partition(&oracle, body["paths"].as_array().unwrap());
    assert_witness_covers_all_vertices(&body, 4);
}

#[tokio::test]
async fn multi_source_multi_sink_dag_uses_three_paths() {
    let run_id = "integration-multi-source-sink-0001";
    let (_typed, input) = input_from_sample("multi-source-multi-sink.json");
    let oracle = enumerate_minimum_path_partitions(&input);

    let (status, body) = post_path_cover(&input, run_id).await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(oracle.minimum_path_count, 3);
    assert_eq!(body["minimum_path_count"], json!(3));
    assert_eq!(body["matching"]["size"], json!(3));
    assert_eq!(
        body["path_count"].as_u64().unwrap() + body["matching"]["size"].as_u64().unwrap(),
        6
    );
    assert_eq!(body["paths"], json!([[0, 2], [1, 3, 5], [4]]));
    assert_concrete_partition(&oracle, body["paths"].as_array().unwrap());
    assert_witness_covers_all_vertices(&body, 6);
}

#[tokio::test]
async fn disconnected_subgraphs_are_solved_without_cross_edges() {
    let run_id = "integration-disconnected-0001";
    let (_typed, input) = input_from_sample("disconnected.json");
    let oracle = enumerate_minimum_path_partitions(&input);

    let (status, body) = post_path_cover(&input, run_id).await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(oracle.minimum_path_count, 3);
    assert_eq!(body["minimum_path_count"], json!(3));
    assert_eq!(body["matching"]["size"], json!(4));
    assert_eq!(
        body["path_count"].as_u64().unwrap() + body["matching"]["size"].as_u64().unwrap(),
        7
    );
    assert_eq!(body["paths"], json!([[0, 1, 2], [3, 4], [5, 6]]));
    assert_concrete_partition(&oracle, body["paths"].as_array().unwrap());
    assert_witness_covers_all_vertices(&body, 7);
}

#[tokio::test]
async fn empty_graph_returns_empty_path_set() {
    let run_id = "integration-empty-graph-0001";
    let input = json!({ "vertices": 0, "edges": [] });
    let oracle = enumerate_minimum_path_partitions(&input);

    let (status, body) = post_path_cover(&input, run_id).await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(oracle.minimum_path_count, 0);
    assert_eq!(body["paths"], json!([]));
    assert_eq!(body["path_count"], json!(0));
    assert_eq!(body["matching"]["size"], json!(0));
    assert_eq!(body["witness"]["vertex_count"], json!(0));
}

#[tokio::test]
async fn cycle_request_fails_with_cycle_evidence_and_classification() {
    let run_id = "integration-cycle-0001";
    let (_typed, input) = input_from_sample("cycle.json");

    let (status, body) = post_path_cover(&input, run_id).await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "body: {body}");
    assert_eq!(body["status"], json!("error"));
    assert_eq!(body["code"], json!("cycle_detected"));
    assert_eq!(body["run_id"], json!(run_id));
    assert_eq!(body["details"]["kind"], json!("cycle_detected"));
    assert_eq!(body["details"]["evidence"]["vertices"], json!([0, 1, 2, 0]));
    assert_eq!(body["details"]["evidence"]["edges"], json!([0, 1, 2]));
    assert!(body.get("paths").is_none());
    assert!(body.get("minimum_path_count").is_none());
}

#[tokio::test]
async fn invalid_edge_fails_as_input_error_not_success() {
    let run_id = "integration-invalid-edge-0001";
    let (_typed, input) = input_from_sample("invalid-edge.json");

    let (status, body) = post_path_cover(&input, run_id).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert_eq!(body["status"], json!("error"));
    assert_eq!(body["code"], json!("invalid_edge_out_of_bounds"));
    assert_eq!(body["run_id"], json!(run_id));
    assert_eq!(
        body["details"]["details"]["edge"],
        json!({ "from": 0, "to": 2 })
    );
    assert!(body.get("paths").is_none());
}

#[tokio::test]
async fn malformed_json_and_unknown_field_keep_error_classification() {
    let run_id = "integration-malformed-json-0001";
    let input = json!({ "vertices": 1, "edges": [], "mystery": true });

    let (status, body) = post_path_cover(&input, run_id).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert_eq!(body["code"], json!("invalid_json"));
    assert_eq!(body["run_id"], json!(run_id));

    let malformed = Value::String("{not-json".to_string());
    let (status, body) = post_path_cover(&malformed, run_id).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body["code"],
        expected_error_json("invalid_json", run_id)["code"]
    );
}

fn assert_witness_covers_all_vertices(body: &Value, n: u64) {
    assert_eq!(body["witness"]["accepted"], json!(true));
    assert_eq!(body["witness"]["vertex_count"], json!(n));

    let mut vertices: Vec<u64> = body["paths"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|path| {
            path.as_array()
                .unwrap()
                .iter()
                .map(|vertex| vertex.as_u64().unwrap())
        })
        .collect();
    vertices.sort_unstable();
    assert_eq!(vertices, (0..n).collect::<Vec<u64>>());
    assert_eq!(vertices.len(), n as usize);
}
