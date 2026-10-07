from __future__ import annotations

import httpx


def _create(client: httpx.Client) -> str:
    response = client.post("/orders", json={})
    assert response.status_code == 201
    return response.json()["id"]


def test_reference_rejects_repeat_close(reference_url: str) -> None:
    with httpx.Client(base_url=reference_url, timeout=3, trust_env=False) as client:
        order_id = _create(client)
        assert client.post(f"/orders/{order_id}/submit").status_code == 200
        assert client.post(f"/orders/{order_id}/close").status_code == 200
        response = client.post(f"/orders/{order_id}/close")
        assert response.status_code == 409
        assert response.json()["category"] == "state_conflict"
        assert client.get(f"/orders/{order_id}").json()["state"] == "closed"


def test_reference_rejects_out_of_order_submit_and_cancel(
    reference_url: str,
) -> None:
    with httpx.Client(base_url=reference_url, timeout=3, trust_env=False) as client:
        draft_id = _create(client)
        close = client.post(f"/orders/{draft_id}/close")
        assert close.status_code == 409
        assert close.json()["category"] == "state_conflict"
        assert client.get(f"/orders/{draft_id}").json()["state"] == "draft"

        submitted_id = _create(client)
        client.post(f"/orders/{submitted_id}/submit")
        client.post(f"/orders/{submitted_id}/close")
        cancel = client.post(f"/orders/{submitted_id}/cancel")
        assert cancel.status_code == 409
        assert cancel.json()["category"] == "state_conflict"
        assert (
            client.get(f"/orders/{submitted_id}").json()["state"] == "closed"
        )


def test_buggy_service_exposes_three_required_state_acceptance_bugs(
    buggy_url: str,
) -> None:
    with httpx.Client(base_url=buggy_url, timeout=3, trust_env=False) as client:
        repeat_id = _create(client)
        client.post(f"/orders/{repeat_id}/submit")
        client.post(f"/orders/{repeat_id}/close")
        repeat = client.post(f"/orders/{repeat_id}/close")
        assert repeat.status_code == 200
        assert repeat.json()["state"] == "closed"

        draft_id = _create(client)
        early_close = client.post(f"/orders/{draft_id}/close")
        assert early_close.status_code == 200
        assert early_close.json()["state"] == "draft"
        assert client.get(f"/orders/{draft_id}").json()["state"] == "draft"

        cancelled_id = _create(client)
        client.post(f"/orders/{cancelled_id}/submit")
        client.post(f"/orders/{cancelled_id}/close")
        late_cancel = client.post(f"/orders/{cancelled_id}/cancel")
        assert late_cancel.status_code == 200
        assert late_cancel.json()["state"] == "closed"
        assert (
            client.get(f"/orders/{cancelled_id}").json()["state"] == "closed"
        )


def test_synthetic_error_categories_remain_distinguishable(
    reference_url: str,
) -> None:
    with httpx.Client(base_url=reference_url, timeout=3, trust_env=False) as client:
        exhausted = client.post(
            "/orders",
            json={},
            headers={"X-Simulate-Fault": "resource_exhausted"},
        )
        assert exhausted.status_code == 507
        assert exhausted.json()["category"] == "resource_exhausted"

        failed = client.post(
            "/orders",
            json={},
            headers={"X-Simulate-Fault": "computation_failure"},
        )
        assert failed.status_code == 500
        assert failed.json()["category"] == "computation_failure"
