from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from milp.app import app
from milp.config import Settings
from milp.errors import ComputationError, StateConflictError
from milp.schemas import SolveOptions
from milp import solver as solver_module


def client() -> TestClient:
    return TestClient(app)


def test_api_solves_and_returns_replay_metadata(settings, monkeypatch):
    monkeypatch.setattr("milp.app.get_settings", lambda: settings)
    response = client().post(
        "/api/solve",
        json={
            "problem": {
                "name": "api-bin",
                "sense": "maximize",
                "objective": {"x": 2, "y": 1},
                "variables": [
                    {"name": "x", "type": "binary"},
                    {"name": "y", "type": "binary"},
                ],
                "constraints": [
                    {"name": "cap", "terms": {"x": 1, "y": 1}, "sense": "<=", "rhs": 1.5}
                ],
            },
            "options": {"run_id": "api-ok"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "optimal"
    assert body["objective_value"] == 2.0
    assert body["absolute_gap"] == 0.0
    assert body["solution"] == {"x": 1.0, "y": 0.0}
    assert body["journal_path"].endswith("api-ok.jsonl")
    assert body["events"]
    branch_event = next(event for event in body["events"] if event["event"] == "branch")
    assert branch_event["branch_variable"] in {"x", "y"}
    assert branch_event["branch_value"] is None or branch_event["branch_value"] >= 0


def test_api_infeasible_response_is_valid_json(settings, monkeypatch):
    monkeypatch.setattr("milp.app.get_settings", lambda: settings)
    response = client().post(
        "/api/solve",
        json={
            "problem": {
                "name": "api-infeasible",
                "sense": "minimize",
                "objective": {"x": 1},
                "variables": [{"name": "x", "type": "integer", "lower": 0, "upper": 1}],
                "constraints": [{"name": "impossible", "terms": {"x": 1}, "sense": ">=", "rhs": 2}],
            },
            "options": {"run_id": "api-infeasible"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "infeasible"
    assert body["events"][0]["node_bound"] is None


def test_api_input_error_is_distinct(settings, monkeypatch):
    monkeypatch.setattr("milp.app.get_settings", lambda: settings)
    response = client().post(
        "/api/solve",
        json={
            "problem": {
                "sense": "minimize",
                "objective": {"missing": 1},
                "variables": [{"name": "x", "type": "binary"}],
            }
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["category"] == "input_error"


def test_api_schema_validation_error_is_input_error(settings, monkeypatch):
    monkeypatch.setattr("milp.app.get_settings", lambda: settings)
    response = client().post("/api/solve", json={"problem": {"name": "bad"}})

    assert response.status_code == 400
    assert response.json()["error"]["category"] == "input_error"


def test_duplicate_run_id_is_state_conflict(settings):
    problem = {
        "name": "dup",
        "sense": "minimize",
        "objective": {"x": 1},
        "variables": [{"name": "x", "type": "binary"}],
    }
    from milp.schemas import LPProblem

    model = LPProblem.model_validate(problem)
    solver_module.solve(model, SolveOptions(run_id="duplicate"), settings=settings)

    with pytest.raises(StateConflictError) as excinfo:
        solver_module.solve(model, SolveOptions(run_id="duplicate"), settings=settings)
    assert excinfo.value.category == "state_conflict"
    assert excinfo.value.http_status == 409


def test_lp_exception_is_computation_failure(settings, monkeypatch):
    def broken_linprog(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(solver_module, "linprog", broken_linprog)
    from milp.schemas import LPProblem

    problem = LPProblem.model_validate(
        {
            "name": "broken",
            "sense": "minimize",
            "objective": {"x": 1},
            "variables": [{"name": "x", "type": "binary"}],
        }
    )

    with pytest.raises(ComputationError) as excinfo:
        solver_module.solve(problem, SolveOptions(run_id="broken"), settings=settings)
    assert excinfo.value.category == "computation_failed"
    assert excinfo.value.http_status == 500


def test_bad_environment_configuration_is_input_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MILP_JOURNAL_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MILP_DEFAULT_NODE_LIMIT", "zero")
    with pytest.raises(Exception) as excinfo:
        Settings.from_env()
    assert excinfo.value.category == "input_error"


def test_oracle_does_not_use_solver():
    from tests import brute_force

    source = open(brute_force.__file__, encoding="utf-8").read()
    assert "from milp.solver" not in source
    assert "import linprog" not in source
