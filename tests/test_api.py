import sqlite3


def _group_by_name(response, name):
    return next(group for group in response["groups"] if group["group"] == name)


def test_health_reports_versions_and_request_identity(client):
    response = client.get("/health", headers={"X-Request-ID": "req-health-1"})
    assert response.status_code == 200
    assert response.json()["request_id"] == "req-health-1"
    assert response.json()["schema_version"] == "survival-analysis.v1"
    assert response.headers["x-request-id"] == "req-health-1"


def test_create_analysis_returns_exact_curve_comparison_failures_and_correlation(client, analysis_payload):
    response = client.post(
        "/api/v1/analyses",
        json=analysis_payload,
        headers={"X-Request-ID": "req-analysis-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-analysis-1"
    assert body["analysis_id"].startswith("km-")
    assert body["kernel_version"] == "km-logrank.v1"
    assert body["config"]["schema_migration_version"] == 1
    assert body["processing_location"].startswith("local-process:")

    group_a = _group_by_name(body, "A")
    assert group_a["n"] == 5
    assert group_a["events"] == 2
    assert group_a["median_survival_time"] is None
    assert group_a["median_status"] == "not_reached"
    assert [(p["time"], p["risk_before"], p["events"], p["censored"], p["survival"]) for p in group_a["points"]] == [
        (0.0, 0, 0, 0, 1.0),
        (2.0, 5, 1, 0, 0.8),
        (4.0, 4, 0, 1, 0.8),
        (5.0, 3, 1, 0, 8 / 15),
        (6.0, 2, 0, 1, 8 / 15),
        (8.0, 1, 0, 1, 8 / 15),
    ]

    empty_group = _group_by_name(body, "EMPTY")
    assert empty_group["n"] == 0
    assert empty_group["points"] == []
    assert empty_group["median_survival_time"] is None
    assert empty_group["median_status"] == "not_estimable"

    assert body["comparison"]["status"] == "failed"
    assert body["comparison"]["reason"] == "empty_group_excluded_from_log_rank"
    assert body["failures"] == [
        {
            "code": "empty_group_excluded_from_log_rank",
            "message": "Log-rank comparison was not produced because the requested comparison is not statistically identifiable.",
            "affects": "comparison:log_rank",
        }
    ]

    trace_components = {event["component"] for event in body["trace"]}
    assert {"api", "risk-kernel", "kaplan-meier", "log-rank", "sqlite"} <= trace_components
    assert "not validated for medical use" in body["disclaimer"].lower()


def test_successful_logrank_request_is_persisted_and_events_are_queryable(client, analysis_payload):
    analysis_payload["groups"] = ["A", "B"]
    created = client.post("/api/v1/analyses", json=analysis_payload)
    body = created.json()
    analysis_id = body["analysis_id"]

    fetched = client.get(f"/api/v1/analyses/{analysis_id}")
    assert fetched.status_code == 200
    assert fetched.json()["comparison"]["status"] == "estimated"
    assert fetched.json()["comparison"]["chi_square"] == 0.4608695652173912
    assert fetched.json()["trace"] == body["trace"]

    events = client.get(f"/api/v1/analyses/{analysis_id}/events")
    assert events.status_code == 200
    records = events.json()["events"]
    assert records[0]["detail"]["request_id"] == body["request_id"]
    assert any(record["component"] == "sqlite" for record in records)
    assert [record["sequence"] for record in records] == list(range(len(records)))


def test_no_events_groups_returns_not_run_without_p_value(client, analysis_payload):
    analysis_payload["groups"] = ["X"]
    analysis_payload["observations"] = [
        {"observation_id": "X1", "group": "X", "time": 1.0, "event": False},
        {"observation_id": "X2", "group": "X", "time": 2.0, "event": False},
    ]
    response = client.post("/api/v1/analyses", json=analysis_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["comparison"]["status"] == "not_run"
    assert body["comparison"]["reason"] == "log_rank_requires_at_least_two_non_empty_groups"
    assert body["comparison"]["p_value"] is None
    assert body["groups"][0]["median_survival_time"] is None
    assert any(notice["code"] == "no_events_median_not_reached" for notice in body["notices"])


def test_unknown_analysis_id_is_404_with_failure_category(client):
    response = client.get("/api/v1/analyses/km-missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ANALYSIS_NOT_FOUND"
    assert response.json()["failures"][0]["affects"] == "request"


def test_validation_errors_list_specific_failure_categories(client, analysis_payload):
    analysis_payload["observations"][0]["entry"] = analysis_payload["observations"][0]["time"]
    response = client.post("/api/v1/analyses", json=analysis_payload)
    assert response.status_code == 422
    detail = response.json()["error"]["details"][0]
    assert detail["location"] == "body.observations.0"
    assert detail["failure_category"] == "ENTRY_NOT_BEFORE_TIME"


def test_duplicate_observation_id_is_rejected(client, analysis_payload):
    analysis_payload["observations"][1]["observation_id"] = "A1"
    response = client.post("/api/v1/analyses", json=analysis_payload)
    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["failure_category"] == "DUPLICATE_OBSERVATION_ID"


def test_http_log_is_written_with_request_id(client):
    client.get("/health", headers={"X-Request-ID": "req-log-1"})
    settings = client.app.state.settings
    with sqlite3.connect(settings.db_path) as connection:
        row = connection.execute(
            "SELECT request_id, status_code, path FROM http_logs WHERE request_id = ?",
            ("req-log-1",),
        ).fetchone()
    assert row == ("req-log-1", 200, "/health")


def test_synthetic_dataset_endpoint_is_reproducible_and_non_medical(client):
    payload = {"seed": 42, "sample_size_per_group": 3, "delayed_entry": True}
    first = client.post("/api/v1/synthetic-dataset", json=payload).json()
    second = client.post("/api/v1/synthetic-dataset", json=payload).json()
    assert first["observations"] == second["observations"]
    assert len(first["observations"]) == 9
    assert first["generator_version"] == "synthetic-equipment-weibull.v1"
    assert "not a real medical" in first["purpose"]
    assert all(item["entry"] < item["time"] for item in first["observations"])


def test_synthetic_endpoint_rejects_reversed_censor_window(client):
    response = client.post(
        "/api/v1/synthetic-dataset",
        json={"seed": 1, "censor_low": 10, "censor_high": 5},
    )
    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["failure_category"] == "INVALID_CENSOR_WINDOW"
