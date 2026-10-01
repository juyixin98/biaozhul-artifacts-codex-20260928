mod support;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use directed_maxflow_backend::api;
use directed_maxflow_backend::api::state::AppState;
use support::{append_log, run_id};

fn valid_request() -> serde_json::Value {
    serde_json::json!({
        "node_count": 4,
        "source": 0,
        "sink": 3,
        "edges": [
            {"from": 0, "to": 1, "capacity": 10},
            {"from": 0, "to": 2, "capacity": 9},
            {"from": 1, "to": 3, "capacity": 7},
            {"from": 2, "to": 3, "capacity": 8},
            {"from": 1, "to": 2, "capacity": 3}
        ]
    })
}

#[tokio::test]
async fn synchronous_compute_returns_checked_flow_and_cut() {
    let app = api::router(AppState::new(4));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/max-flow/compute")
                .header("content-type", "application/json")
                .body(Body::from(valid_request().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["flow_value"], 15);
    assert_eq!(body["certificate"]["status"], "proved");
    assert_eq!(body["certificate"]["cut_check"]["cut_capacity"], 15);
    assert_eq!(
        body["certificate"]["cut_check"]["source_side"],
        serde_json::json!([0, 1, 2])
    );
    assert_eq!(
        body["certificate"]["cut_check"]["sink_side"],
        serde_json::json!([3])
    );
    assert_eq!(
        body["certificate"]["flow_check"]["balances"],
        serde_json::json!([15, 0, 0, -15])
    );
    assert!(body["progress"].as_array().unwrap().len() >= 3);
}

#[tokio::test]
async fn input_errors_are_distinct_from_state_and_resource_errors() {
    let app = api::router(AppState::new(4));
    let mut invalid = valid_request();
    invalid["source"] = 3;
    invalid["sink"] = 3;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/max-flow/compute")
                .header("content-type", "application/json")
                .body(Body::from(invalid.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = read_json(response).await;
    assert_eq!(body["category"], "input_error");
    assert_eq!(body["error"]["kind"], "source_equals_sink");
    assert!(body["run_id"].is_string());

    let negative = serde_json::json!({
        "node_count": 2,
        "source": 0,
        "sink": 1,
        "edges": [{"from": 0, "to": 1, "capacity": -3}]
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/max-flow/compute")
                .header("content-type", "application/json")
                .body(Body::from(negative.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = read_json(response).await;
    assert_eq!(body["category"], "input_error");
    assert_eq!(body["error"]["kind"], "malformed_json");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/max-flow/jobs/missing/start")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = read_json(response).await;
    assert_eq!(body["category"], "not_found");
    assert_eq!(body["run_id"], "missing");
}

#[tokio::test]
async fn queued_job_can_be_canceled_before_start_and_terminal_cancel_conflicts() {
    let state = AppState::new(4);
    let app = api::router(state.clone());

    let mut request = valid_request();
    request["autostart"] = false;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/max-flow/jobs")
                .header("content-type", "application/json")
                .body(Body::from(request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let body = read_json(response).await;
    assert_eq!(body["status"], "queued");
    let run_id = body["run_id"].as_str().unwrap().to_string();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/max-flow/jobs/{run_id}/cancel"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = read_json(response).await;
    assert_eq!(body["status"], "canceled");
    assert_eq!(body["flow_value"], 0);
    assert_eq!(body["certificate"]["status"], "not_proved");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/max-flow/jobs/{run_id}/cancel"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let body = read_json(response).await;
    assert_eq!(body["category"], "state_conflict");

    let log_run = run_id("api", "queued_cancel");
    append_log(
        &log_run,
        "state_transition",
        "queued -> canceled returned the feasible zero flow; canceling again is a terminal state conflict",
        &serde_json::json!({"job_run_id": run_id, "response": body}),
    );
}

#[tokio::test]
async fn running_job_cancellation_returns_current_feasible_flow() {
    let state = AppState::new(4);
    let app = api::router(state.clone());
    let mut request = lattice_request(100, 100);
    request["autostart"] = false;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/max-flow/jobs")
                .header("content-type", "application/json")
                .body(Body::from(request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let body = read_json(response).await;
    let run_id = body["run_id"].as_str().unwrap().to_string();

    state.jobs.start(&run_id).unwrap();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/max-flow/jobs/{run_id}/cancel"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(response.status() == StatusCode::ACCEPTED || response.status() == StatusCode::OK);

    let mut final_body = None;
    for _ in 0..100 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("/max-flow/jobs/{run_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = read_json(response).await;
        if body["status"] == "canceled" {
            final_body = Some(body);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    let body = final_body.expect("job reaches canceled state");
    assert!(body["flow_value"].as_u64().unwrap() <= 100);
    assert_eq!(body["certificate"]["status"], "not_proved");
    assert!(body["certificate"]["cut_check"].is_null());
    assert_eq!(
        body["certificate"]["flow_check"]["flow_value"],
        body["flow_value"]
    );

    let log_run = run_id("api", "running_cancel");
    append_log(
        &log_run,
        "cancellation",
        "running cancellation finalized with current feasible flow 4 and explicitly omitted the optimality cut",
        &serde_json::json!({"job_run_id": run_id, "final_response": body}),
    );
}

fn lattice_request(rows: usize, cols: usize) -> serde_json::Value {
    let node_id = |row: usize, col: usize| row * cols + col;
    let mut edges = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let node = node_id(row, col);
            if row + 1 < rows {
                edges.push(serde_json::json!({
                    "from": node,
                    "to": node_id(row + 1, col),
                    "capacity": 1
                }));
            }
            if col + 1 < cols {
                edges.push(serde_json::json!({
                    "from": node,
                    "to": node_id(row, col + 1),
                    "capacity": 1
                }));
            }
        }
    }
    serde_json::json!({
        "node_count": rows * cols,
        "source": 0,
        "sink": rows * cols - 1,
        "edges": edges
    })
}

#[tokio::test]
async fn resource_exhaustion_and_computation_failure_remain_distinct() {
    let app = api::router(AppState::new(2));
    let mut queued = valid_request();
    queued["autostart"] = false;
    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/max-flow/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(queued.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
    }
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/max-flow/jobs")
                .header("content-type", "application/json")
                .body(Body::from(queued.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INSUFFICIENT_STORAGE);
    let body = read_json(response).await;
    assert_eq!(body["category"], "resource_exhausted");
    assert_eq!(body["error"]["kind"], "resource_exhausted");

    let overflow = serde_json::json!({
        "node_count": 4,
        "source": 0,
        "sink": 3,
        "edges": [
            {"from": 0, "to": 1, "capacity": u64::MAX},
            {"from": 0, "to": 2, "capacity": u64::MAX},
            {"from": 1, "to": 3, "capacity": u64::MAX},
            {"from": 2, "to": 3, "capacity": u64::MAX}
        ]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/max-flow/compute")
                .header("content-type", "application/json")
                .body(Body::from(overflow.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = read_json(response).await;
    assert_eq!(body["category"], "computation_failed");
    assert_eq!(body["error"]["kind"], "arithmetic_overflow");
}

async fn read_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
