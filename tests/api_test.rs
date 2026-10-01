use axum::body::Body;
use axum::http::{Request, StatusCode};
use chordal_api::api::{router, AppState};
use serde_json::{json, Value};
use tower::ServiceExt;

async fn call_json(
    method: &str,
    path: &str,
    payload: Value,
    state_name: &str,
) -> (StatusCode, Value) {
    let state = AppState::new(&format!("test-logs/api/{state_name}.jsonl"));
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("request");
    let response = router(state).oneshot(request).await.expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body");
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

#[tokio::test]
async fn health_endpoint_reports_service_version() {
    let (status, body) = call_json("GET", "/health", json!({}), "health").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], json!("ok"));
    assert_eq!(body["service"], json!("chordal-api"));
    assert_eq!(body["version"], json!("0.1.0"));
}

#[tokio::test]
async fn analyze_returns_full_c4_witness_and_check_reports() {
    let payload = json!({
        "run_id": "api-c4",
        "graph": {
            "vertices": ["a", "b", "c", "d"],
            "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"]]
        }
    });
    let (status, body) = call_json("POST", "/api/chordality/analyze", payload, "c4").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["run_id"], json!("api-c4"));
    let result = &body["result"];
    assert_eq!(result["chordal"], json!(false));
    assert_eq!(
        result["perfect_elimination_order"]["first_failure"]["failure_category"],
        json!("successor_neighbors_not_clique")
    );
    assert_eq!(result["chordless_cycle_witness"]["cycle_length"], json!(4));
    assert_eq!(result["chordless_cycle_witness"]["induced"], json!(true));
    assert_eq!(
        result["chordless_cycle_witness"]["independent_check"]["valid"],
        json!(true)
    );
    assert!(result["trace"].is_array());
}

#[tokio::test]
async fn analyze_normalizes_reversed_and_duplicate_edges() {
    let payload = json!({
        "run_id": "api-duplicates",
        "graph": {
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "a"], ["a", "b"], {"from":"b", "to":"c"}]
        }
    });
    let (status, body) = call_json("POST", "/api/chordality/analyze", payload, "duplicates").await;

    assert_eq!(status, StatusCode::OK);
    let graph = &body["result"]["normalized_graph"];
    assert_eq!(graph["vertex_count"], json!(3));
    assert_eq!(graph["edge_count"], json!(2));
    assert_eq!(graph["duplicate_edge_count"], json!(2));
    assert_eq!(graph["normalized_edges"], json!([["a", "b"], ["b", "c"]]));
}

#[tokio::test]
async fn verify_order_rejects_tampered_order_with_missing_edge() {
    let payload = json!({
        "run_id": "api-tampered",
        "graph": {
            "vertices": ["a", "b", "c", "d"],
            "edges": [["a", "b"], ["a", "c"], ["b", "c"], ["b", "d"], ["c", "d"]]
        },
        "order": ["b", "a", "d", "c"]
    });
    let (status, body) =
        call_json("POST", "/api/chordality/verify-order", payload, "tampered").await;

    assert_eq!(status, StatusCode::OK);
    let check = &body["result"]["order_check"];
    assert_eq!(check["valid"], json!(false));
    assert_eq!(check["first_failure"]["vertex"], json!("b"));
    assert_eq!(check["first_failure"]["position"], json!(0));
    assert_eq!(
        check["first_failure"]["later_neighbors"],
        json!(["a", "c", "d"])
    );
    assert_eq!(check["first_failure"]["missing_edges"], json!([["a", "d"]]));
}

#[tokio::test]
async fn malformed_json_is_input_error() {
    let state = AppState::new("test-logs/api/malformed.jsonl");
    let request = Request::builder()
        .method("POST")
        .uri("/api/chordality/analyze")
        .header("content-type", "application/json")
        .body(Body::from("{not-json"))
        .unwrap();
    let response = router(state).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["kind"], json!("input_error"));
    assert_eq!(body["error"]["code"], json!("invalid_json"));
    assert!(body["error"]["run_id"].is_string());
}

#[tokio::test]
async fn duplicate_vertex_declaration_is_state_conflict() {
    let payload = json!({
        "run_id": "api-duplicate-vertex",
        "graph": {"vertices": ["a", "a"], "edges": []}
    });
    let (status, body) = call_json(
        "POST",
        "/api/chordality/analyze",
        payload,
        "duplicate-vertex",
    )
    .await;

    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["kind"], json!("state_conflict"));
    assert_eq!(body["error"]["code"], json!("duplicate_vertex_id"));
}

#[tokio::test]
async fn self_loop_is_input_error() {
    let payload = json!({
        "run_id": "api-self-loop",
        "graph": {"edges": [["a", "a"]]}
    });
    let (status, body) = call_json("POST", "/api/chordality/analyze", payload, "self-loop").await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["kind"], json!("input_error"));
    assert_eq!(body["error"]["code"], json!("invalid_graph"));
}

#[tokio::test]
async fn too_many_vertices_is_resource_exhausted() {
    let vertices: Vec<String> = (0..129).map(|value| value.to_string()).collect();
    let payload = json!({
        "run_id": "api-too-large",
        "graph": {"vertices": vertices, "edges": []}
    });
    let (status, body) = call_json("POST", "/api/chordality/analyze", payload, "too-large").await;

    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(body["error"]["kind"], json!("resource_exhausted"));
    assert_eq!(body["error"]["code"], json!("graph_too_large"));
}

#[tokio::test]
async fn unknown_route_is_not_found() {
    let (status, body) = call_json("GET", "/nope", json!({}), "not-found").await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["kind"], json!("not_found"));
    assert_eq!(body["error"]["code"], json!("route_not_found"));
}
