mod common;

use axum::body::Body;
use axum::http::{HeaderName, HeaderValue, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;
use truss_backend::{api::router, config::Settings};

async fn call_json(
    settings: Settings,
    body: Value,
    uri: &str,
    request_id: Option<&str>,
) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .header(
            "x-request-id",
            HeaderValue::from_str(request_id.unwrap_or("test-request-001")).unwrap(),
        )
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = router(settings).oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_parts().1, usize::MAX)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    (status, value)
}

#[tokio::test]
async fn analyze_accepts_fixture_and_returns_concrete_witness_data() {
    let (status, body) = call_json(
        Settings::default(),
        common::load_fixture_json("clique_bridge.json"),
        "/api/v1/graphs/analyze",
        Some("clique-bridge-1"),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["request_id"], json!("clique-bridge-1"));
    assert_eq!(body["disposition"], json!("accepted"));
    assert_eq!(body["verification_accepted"], json!(true));
    assert_eq!(body["triangle_count"], json!(5));
    assert_eq!(body["edges"].as_array().unwrap().len(), 10);

    let bridge = body["edges"]
        .as_array()
        .unwrap()
        .iter()
        .find(|edge| edge["edge_id"] == json!("bridge_dx"))
        .unwrap();
    assert_eq!(bridge["level"], json!(2));
    assert_eq!(bridge["initial_support"], json!(0));

    let k4_edge = body["edges"]
        .as_array()
        .unwrap()
        .iter()
        .find(|edge| edge["edge_id"] == json!("k4_ab"))
        .unwrap();
    assert_eq!(k4_edge["level"], json!(4));
    assert_eq!(k4_edge["initial_support"], json!(2));

    let witnesses = body["witnesses"].as_array().unwrap();
    assert!(witnesses.iter().any(|witness| {
        witness["level"] == json!(4)
            && witness["edge_count"] == json!(6)
            && witness["minimum_support"] == json!(2)
            && witness["triangle_count"] == json!(4)
    }));
    assert!(witnesses.iter().any(|witness| {
        witness["level"] == json!(3)
            && witness["edge_count"] == json!(3)
            && witness["minimum_support"] == json!(1)
            && witness["triangle_count"] == json!(1)
    }));
    assert!(witnesses.iter().any(|witness| {
        let edges = witness["edge_ids"].as_array().unwrap();
        witness["level"] == json!(2)
            && witness["minimum_support"] == json!(0)
            && edges.iter().any(|edge| *edge == json!("bridge_dx"))
    }));
}

#[tokio::test]
async fn analyze_rejects_self_loops_with_status_code_and_diagnostic_code() {
    let (status, body) = call_json(
        Settings::default(),
        common::load_fixture_json("invalid_self_loop.json"),
        "/api/v1/graphs/analyze",
        Some("bad-loop"),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["request_id"], json!("bad-loop"));
    assert_eq!(body["disposition"], json!("rejected"));
    assert_eq!(body["code"], json!("INVALID_GRAPH"));
    assert_eq!(body["diagnostics"][0]["code"], json!("SELF_LOOP"));
    assert_eq!(body["diagnostics"][0]["state"]["edge_id"], json!("loop"));
    assert!(
        body["diagnostics"][0]["state"]["redacted_endpoint"]
            .as_str()
            .unwrap()
            .starts_with("id_")
    );
}

#[tokio::test]
async fn malformed_json_is_a_400_bad_request() {
    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/graphs/analyze")
        .header("x-request-id", "bad-json")
        .body(Body::from("{not-json"))
        .unwrap();
    let response = router(Settings::default()).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_parts().1, usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["request_id"], json!("bad-json"));
    assert_eq!(body["code"], json!("MALFORMED_REQUEST"));
}

#[tokio::test]
async fn multiple_edge_policy_can_reject_duplicate_declarations() {
    let mut body = common::load_fixture_json("duplicate_edges.json");
    body["multiple_edge_policy"] = json!("reject");

    let (status, response) = call_json(
        Settings::default(),
        body,
        "/api/v1/graphs/analyze",
        Some("duplicate-policy"),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(response["diagnostics"][0]["code"], json!("MULTIPLE_EDGE"));
    assert_eq!(
        response["diagnostics"][0]["state"]["declared_count"],
        json!("2")
    );
}

#[tokio::test]
async fn edge_query_projects_one_edge_and_its_witness() {
    let (status, body) = call_json(
        Settings::default(),
        common::load_fixture_json("clique_bridge.json"),
        "/api/v1/graphs/analyze?edge_id=k4_ab",
        Some("one-edge"),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["edges"].as_array().unwrap().len(), 1);
    assert_eq!(body["edges"][0]["edge_id"], json!("k4_ab"));
    assert_eq!(body["witnesses"].as_array().unwrap().len(), 1);
    assert_eq!(body["witnesses"][0]["witness_id"], json!("k4-c0"));
}

#[tokio::test]
async fn missing_edge_query_is_rejected_with_404() {
    let (status, body) = call_json(
        Settings::default(),
        common::load_fixture_json("zero_triangle.json"),
        "/api/v1/graphs/analyze?edge_id=absent",
        Some("missing-edge"),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], json!("EDGE_NOT_FOUND"));
    assert_eq!(body["request_id"], json!("missing-edge"));
}

#[tokio::test]
async fn configured_size_limit_reports_a_specific_failure_class() {
    let settings = Settings {
        max_nodes: 1,
        ..Default::default()
    };
    let (status, body) = call_json(
        settings,
        common::load_fixture_json("zero_triangle.json"),
        "/api/v1/graphs/analyze",
        Some("too-large"),
    )
    .await;

    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(body["code"], json!("NODE_LIMIT_EXCEEDED"));
    assert_eq!(
        body["diagnostics"][0]["state"]["configured_limit"],
        json!("1")
    );
}

#[tokio::test]
async fn health_endpoint_is_available() {
    let response = router(Settings::default())
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_parts().1, usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["status"], json!("ok"));
}

#[test]
fn request_id_header_name_is_http_safe() {
    assert_eq!(
        HeaderName::from_static("x-request-id").as_str(),
        "x-request-id"
    );
}
