use axum::body::Body;
use axum::http::{Request, StatusCode};
use msf_service::api::{router, AppState};
use msf_service::config::AppConfig;
use msf_service::store::GraphStore;
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;

fn test_config() -> AppConfig {
    AppConfig {
        max_history_versions: 3,
        ..AppConfig::default()
    }
}

async fn request_json(
    app: &axum::Router,
    method: &str,
    uri: &str,
    body: Value,
    request_id: &str,
) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-request-id", request_id)
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(
        response
            .headers()
            .get("x-request-id")
            .unwrap()
            .to_str()
            .unwrap(),
        request_id
    );
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        panic!(
            "status {status}, body {}, parse error {error}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, value)
}

async fn create_graph(app: &axum::Router, id: &str, request_id: &str) -> (StatusCode, Value) {
    request_json(
        app,
        "POST",
        "/graphs",
        json!({ "graph_id": id }),
        request_id,
    )
    .await
}

async fn batch(app: &axum::Router, id: &str, body: Value, request_id: &str) -> (StatusCode, Value) {
    request_json(
        app,
        "POST",
        &format!("/graphs/{id}/batch-updates"),
        body,
        request_id,
    )
    .await
}

#[tokio::test]
async fn batch_publishes_only_one_complete_version_and_returns_steps() {
    let app = router(AppState {
        store: Arc::new(GraphStore::new()),
        config: test_config(),
    });
    let (status, created) = create_graph(&app, "g1", "create-1").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(created["request_id"], json!("create-1"));
    assert_eq!(created["data"]["version"], json!(0));

    let body = json!({
        "expected_version": 0,
        "operations": [
            { "op": "add_edge", "edge": { "id": "a", "u": 1, "v": 2, "weight": 4 } },
            { "op": "add_edge", "edge": { "id": "b", "u": 2, "v": 3, "weight": 2 } },
            { "op": "add_edge", "edge": { "id": "loop", "u": 3, "v": 3, "weight": -100 } },
            { "op": "set_weight", "edge_id": "a", "weight": 1 }
        ]
    });
    let (status, response) = batch(&app, "g1", body, "batch-1").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["request_id"], json!("batch-1"));
    assert_eq!(response["data"]["previous_version"], json!(0));
    assert_eq!(response["data"]["published_version"], json!(1));
    assert_eq!(response["data"]["snapshot"]["version"], json!(1));
    assert_eq!(response["data"]["snapshot"]["total_weight"], json!("3"));
    assert_eq!(
        response["data"]["snapshot"]["forest_edge_ids"],
        json!(["a", "b"])
    );
    assert_eq!(
        response["data"]["steps"][0]["action"],
        json!("added_to_forest")
    );
    assert_eq!(
        response["data"]["steps"][2]["action"],
        json!("ignored_self_loop")
    );
    assert_eq!(
        response["data"]["steps"][3]["action"],
        json!("tree_weight_changed_still_selected")
    );
    assert_eq!(response["data"]["witness"]["status"], json!("accepted"));
}

#[tokio::test]
async fn expected_version_conflict_rejects_batch_and_does_not_publish() {
    let app = router(AppState {
        store: Arc::new(GraphStore::new()),
        config: test_config(),
    });
    create_graph(&app, "g2", "create-2").await;

    let first = json!({
        "expected_version": 0,
        "operations": [
            { "op": "add_edge", "edge": { "id": "a", "u": 1, "v": 2, "weight": 1 } }
        ]
    });
    let (status, _) = batch(&app, "g2", first, "first-write").await;
    assert_eq!(status, StatusCode::OK);

    let stale = json!({
        "expected_version": 0,
        "operations": [
            { "op": "add_edge", "edge": { "id": "b", "u": 2, "v": 3, "weight": 2 } }
        ]
    });
    let (status, response) = batch(&app, "g2", stale, "stale-write").await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(
        response["error"]["code"],
        json!("expected_version_mismatch")
    );
    assert_eq!(response["request_id"], json!("stale-write"));

    let request = Request::builder()
        .uri("/graphs/g2/versions")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["data"]["versions"], json!([0, 1]));
}

#[tokio::test]
async fn invalid_batch_fails_with_specific_error_and_keeps_previous_version() {
    let app = router(AppState {
        store: Arc::new(GraphStore::new()),
        config: test_config(),
    });
    create_graph(&app, "g3", "create-3").await;

    let body = json!({
        "expected_version": 0,
        "operations": [
            { "op": "add_edge", "edge": { "id": "a", "u": 1, "v": 2, "weight": 1 } },
            { "op": "delete_edge", "edge_id": "missing" }
        ]
    });
    let (status, response) = batch(&app, "g3", body, "bad-batch").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(response["error"]["code"], json!("edge_not_found"));
    assert_eq!(
        response["error"]["location"],
        json!("operations[].edge_id=missing")
    );

    let request = Request::builder()
        .uri("/graphs/g3/versions/1")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn history_retention_keeps_configured_number_of_complete_versions() {
    let app = router(AppState {
        store: Arc::new(GraphStore::new()),
        config: test_config(),
    });
    create_graph(&app, "g4", "create-4").await;

    for version in 1..=4u64 {
        let body = json!({
            "expected_version": version - 1,
            "operations": [
                { "op": "add_edge", "edge": { "id": format!("e{version}"), "u": version as i64, "v": version as i64 + 1, "weight": version as i64 } }
            ]
        });
        let (status, _) = batch(&app, "g4", body, &format!("write-{version}")).await;
        assert_eq!(status, StatusCode::OK);
    }

    let old_request = Request::builder()
        .uri("/graphs/g4/versions/1")
        .body(Body::empty())
        .unwrap();
    let old = app.clone().oneshot(old_request).await.unwrap();
    assert_eq!(old.status(), StatusCode::NOT_FOUND);

    let latest_request = Request::builder()
        .uri("/graphs/g4/versions/4")
        .body(Body::empty())
        .unwrap();
    let latest = app.oneshot(latest_request).await.unwrap();
    assert_eq!(latest.status(), StatusCode::OK);
}

#[tokio::test]
async fn witness_endpoint_returns_concrete_failure_category() {
    let body = json!({
        "claim": {
            "vertices": [1, 2],
            "edges": {
                "a": { "u": 1, "v": 2, "weight": 1 },
                "loop": { "u": 2, "v": 2, "weight": -9 }
            },
            "forest_edge_ids": ["loop"]
        }
    });
    let app = router(AppState {
        store: Arc::new(GraphStore::new()),
        config: test_config(),
    });
    let (status, response) =
        request_json(&app, "POST", "/witness-checks", body, "witness-bad").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["data"]["status"], json!("rejected"));
    assert_eq!(
        response["data"]["failures"][0]["category"],
        json!("forest_edge_is_self_loop")
    );
}
