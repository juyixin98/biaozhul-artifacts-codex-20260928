use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use maxflow_server::api;
use tower::ServiceExt;

fn app() -> Router {
    api::router()
}

async fn response_json(
    application: Router,
    request: Request<Body>,
) -> (StatusCode, serde_json::Value) {
    let response = application.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap();
    let value = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        panic!(
            "response body was JSON: {error}; body={}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, value)
}

async fn wait_for_job(application: Router, job_id: &str) -> serde_json::Value {
    for _ in 0..100 {
        let (_, body) = response_json(application.clone(), get(&format!("/jobs/{job_id}"))).await;
        if body["state"] != "running" {
            return body;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("job {job_id} did not finish");
}

fn post(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn get(path: &str) -> Request<Body> {
    Request::get(path).body(Body::empty()).unwrap()
}

#[tokio::test]
async fn completes_job_and_returns_flow_conservation_and_cut() {
    let application = app();
    let request = post(
        "/jobs",
        serde_json::json!({
            "node_count": 4,
            "source": 0,
            "sink": 3,
            "edges": [
                {"id": "sa", "from": 0, "to": 1, "capacity": 40},
                {"id": "sb", "from": 0, "to": 2, "capacity": 30},
                {"id": "ba", "from": 2, "to": 1, "capacity": 20},
                {"id": "at", "from": 1, "to": 3, "capacity": 10},
            ]
        }),
    );
    let (status, body) = response_json(application.clone(), request).await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = body["id"].as_str().unwrap().to_string();

    let body = wait_for_job(application, &job_id).await;
    assert_eq!(body["state"], "completed");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["optimal"], true);
    assert_eq!(body["certificate"]["value"], 10);
    assert_eq!(
        body["certificate"]["cut"]["source_side"],
        serde_json::json!([0, 1, 2])
    );
    assert_eq!(
        body["certificate"]["cut"]["sink_side"],
        serde_json::json!([3])
    );
    assert_eq!(body["feasibility_valid"], true);
    assert!(body["events"].as_array().unwrap().len() >= 3);
    assert_eq!(
        body["run_id"].as_str().unwrap(),
        body["events"][0]["run_id"]
    );
}

#[tokio::test]
async fn phase_budget_cancels_with_feasible_flow_and_without_optimal_cut() {
    let application = app();
    let (status, body) = response_json(
        application.clone(),
        post(
            "/jobs",
            serde_json::json!({
                "node_count": 4,
                "source": 0,
                "sink": 3,
                "max_phases": 1,
                "edges": [
                    {"id": "sv", "from": 0, "to": 2, "capacity": 10},
                    {"id": "su", "from": 0, "to": 1, "capacity": 10},
                    {"id": "vu", "from": 2, "to": 1, "capacity": 10},
                    {"id": "ut", "from": 1, "to": 3, "capacity": 10},
                    {"id": "vt", "from": 2, "to": 3, "capacity": 10}
                ]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = body["id"].as_str().unwrap();

    let body = wait_for_job(application.clone(), &job_id).await;
    assert_eq!(body["state"], "canceled");
    assert_eq!(body["optimal"], false);
    assert_eq!(body["certificate"]["value"], 20);
    assert!(body["certificate"]["cut"].is_null());
    assert_eq!(body["feasibility_valid"], true);

    let (status, body) = response_json(
        application.clone(),
        Request::post(format!("/jobs/{job_id}/cancel"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["category"], "state_conflict");
    assert_eq!(body["code"], "state_conflict");
}

#[tokio::test]
async fn negative_capacity_is_an_input_error_not_a_computation_error() {
    let (status, body) = response_json(
        app(),
        post(
            "/jobs",
            serde_json::json!({
                "node_count": 2,
                "source": 0,
                "sink": 1,
                "edges": [{"id": "bad", "from": 0, "to": 1, "capacity": -1}]
            }),
        ),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["category"], "input_error");
    assert_eq!(body["code"], "capacity_out_of_range");
}

#[tokio::test]
async fn malformed_json_and_unknown_jobs_have_distinct_categories() {
    let request = Request::post("/jobs")
        .header("content-type", "application/json")
        .body(Body::from("{".to_string()))
        .unwrap();
    let (status, body) = response_json(app(), request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["category"], "input_error");
    assert_eq!(body["code"], "invalid_json");

    let (status, body) = response_json(app(), get("/jobs/missing")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["category"], "not_found");
}

#[tokio::test]
async fn oversized_graph_is_resource_exhausted() {
    let request = post(
        "/jobs",
        serde_json::json!({
            "node_count": 1025,
            "source": 0,
            "sink": 1024,
            "edges": []
        }),
    );
    let (status, body) = response_json(app(), request).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["category"], "resource_exhausted");
    assert_eq!(body["code"], "graph_too_large");
}

#[tokio::test]
async fn witness_endpoint_accepts_valid_certificate_and_rejects_tampered_cut() {
    let valid = serde_json::json!({
        "node_count": 4,
        "source": 0,
        "sink": 3,
        "edges": [
            {"id": "sa", "from": 0, "to": 1, "capacity": 40},
            {"id": "sb", "from": 0, "to": 2, "capacity": 30},
            {"id": "ba", "from": 2, "to": 1, "capacity": 20},
            {"id": "at", "from": 1, "to": 3, "capacity": 10}
        ],
        "value": 10,
        "edge_flows": [
            {"edge_id": "sa", "flow": 10},
            {"edge_id": "sb", "flow": 0},
            {"edge_id": "ba", "flow": 0},
            {"edge_id": "at", "flow": 10}
        ],
        "cut": {"source_side": [0, 1, 2], "sink_side": [3]}
    });
    let (status, body) = response_json(app(), post("/witness", valid)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["valid"], true);

    let mut invalid_request = serde_json::json!({
        "node_count": 4,
        "source": 0,
        "sink": 3,
        "edges": [
            {"id": "sa", "from": 0, "to": 1, "capacity": 40},
            {"id": "sb", "from": 0, "to": 2, "capacity": 30},
            {"id": "at", "from": 1, "to": 3, "capacity": 10},
            {"id": "bt", "from": 2, "to": 3, "capacity": 60}
        ],
        "value": 10,
        "edge_flows": [
            {"edge_id": "sa", "flow": 10},
            {"edge_id": "sb", "flow": 0},
            {"edge_id": "at", "flow": 10},
            {"edge_id": "bt", "flow": 0}
        ],
        "cut": {"source_side": [0, 1, 2], "sink_side": [3]}
    });
    invalid_request["value"] = serde_json::json!(9);
    let (status, body) = response_json(app(), post("/witness", invalid_request)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["valid"], false);
    assert!(body["failures"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str().unwrap().contains("declared value")));
}
