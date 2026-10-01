from __future__ import annotations

from fastapi.testclient import TestClient

from lasso_backend.app import app


client = TestClient(app)


def test_pydantic_contract_rejection_is_classified():
    response = client.post(
        "/api/v1/lasso/path",
        json={"x": [[1.0, 2.0]], "y": [1.0], "lambdas": [0.0]},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["decision"] == "rejected"
    assert body["error_category"] == "invalid_request_contract"
    assert body["details"]
    assert isinstance(body["request_id"], str)
