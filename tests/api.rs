use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use dag_path_cover::{api::router, config::Config};
use serde_json::{json, Value};
use tower::util::ServiceExt;

fn fixture_value(name: &str) -> Value {
    let path = format!(
        concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/{0}.json"),
        name
    );
    let text = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&text).unwrap()
}

async fn post_json(payload: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri("/api/path-cover")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();
    let response = router(Config::default()).oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    (status, value)
}

#[tokio::test]
async fn health_reports_version_and_local_configuration() {
    let request = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let response = router(Config::default()).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["ok"], true);
    assert_eq!(body["service"], "dag-path-cover");
    assert_eq!(body["version"], "0.1.0");
    assert_eq!(body["bind_address"], "127.0.0.1:8080");
}

#[tokio::test]
async fn fixture_listing_and_detail_return_embedded_documents() {
    let request = Request::builder()
        .uri("/api/fixtures")
        .body(Body::empty())
        .unwrap();
    let response = router(Config::default()).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["fixtures"].as_array().unwrap().len(), 5);
    assert!(body.to_string().contains("disconnected_components"));

    let request = Request::builder()
        .uri("/api/fixtures/cycle")
        .body(Body::empty())
        .unwrap();
    let response = router(Config::default()).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["fixture"]["name"], "cycle");
}

#[tokio::test]
async fn post_analyzes_diamond_with_diagnostic_identity_and_steps() {
    let fixture = fixture_value("diamond");
    let payload = json!({
        "case_id": "api-diamond",
        "graph": fixture["graph"]
    });
    let (status, body) = post_json(payload).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["ok"], true);
    let result = &body["result"];
    assert_eq!(result["case_id"], "api-diamond");
    assert_eq!(result["version"], "0.1.0");
    assert_eq!(result["vertex_count"], 4);
    assert_eq!(result["edge_count"], 4);
    assert_eq!(result["minimum_path_count"], 2);
    assert_eq!(result["matching_size"], 2);
    assert_eq!(result["paths"], json!([["s", "a", "t"], ["b"]]));
    assert!(result["run_id"].as_str().unwrap().starts_with("run-"));
    assert!(result["input_id"].as_str().unwrap().starts_with("in-"));
    assert_eq!(result["diagnostics"]["valid_witness"], true);
    let steps = result["diagnostics"]["steps"].as_array().unwrap();
    assert_eq!(steps.first().unwrap()["name"], "receive");
    assert_eq!(steps.last().unwrap()["name"], "witness_check");
}

#[tokio::test]
async fn post_zero_edge_graph_returns_singletons() {
    let fixture = fixture_value("zero_edges");
    let payload = json!({ "graph": fixture["graph"] });
    let (status, body) = post_json(payload).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["result"]["minimum_path_count"], 3);
    assert_eq!(body["result"]["matching_size"], 0);
    assert_eq!(body["result"]["paths"], json!([["a"], ["b"], ["c"]]));
}

#[tokio::test]
async fn post_cycle_is_failure_with_cycle_evidence_and_steps() {
    let fixture = fixture_value("cycle");
    let payload = json!({
        "case_id": "api-cycle",
        "graph": fixture["graph"]
    });
    let (status, body) = post_json(payload).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["ok"], false);
    let error = &body["error"];
    assert_eq!(error["category"], "CYCLE_DETECTED");
    assert_eq!(error["case_id"], "api-cycle");
    assert_eq!(error["details"]["cycle_vertices"], json!(["a", "b", "c"]));
    assert_eq!(
        error["details"]["cycle_edges"],
        json!([
            {"from": "a", "to": "b"},
            {"from": "b", "to": "c"},
            {"from": "c", "to": "a"}
        ])
    );
    assert!(error["steps"]
        .as_array()
        .unwrap()
        .iter()
        .any(|step| step["name"] == "cycle_check" && step["status"] == "failed"));
}

#[tokio::test]
async fn malformed_json_and_unknown_routes_do_not_report_success() {
    let request = Request::builder()
        .method("POST")
        .uri("/api/path-cover")
        .header("content-type", "application/json")
        .body(Body::from("{".to_string()))
        .unwrap();
    let response = router(Config::default()).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["ok"], false);
    assert_eq!(body["error"]["category"], "INVALID_JSON");

    let request = Request::builder().uri("/nope").body(Body::empty()).unwrap();
    let response = router(Config::default()).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["category"], "ROUTE_NOT_FOUND");
}

#[tokio::test]
async fn wrong_content_type_is_rejected() {
    let request = Request::builder()
        .method("POST")
        .uri("/api/path-cover")
        .header("content-type", "text/plain")
        .body(Body::from("{}".to_string()))
        .unwrap();
    let response = router(Config::default()).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["category"], "UNSUPPORTED_MEDIA_TYPE");
}
