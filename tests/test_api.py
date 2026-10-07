"""Report/control API contract tests (self-contained event loop)."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from mdbt.api import create_app


@pytest.fixture
def paths(tmp_path):
    return str(tmp_path / "evidence.db"), str(tmp_path / "replays")


async def _scenario(db, replays):
    app = create_app()
    model = json.loads(open("fixtures/order_machine.json").read())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api"
    ) as client:
        r = await client.get("/healthz")
        assert r.status_code == 200
        assert set(r.json()["suites"]) == {"buggy", "reference"}

        r = await client.post("/models/validate", json={"model": model})
        assert r.status_code == 200 and r.json()["valid"] is True

        bad = dict(model)
        bad["initial_state"] = "nope"
        r = await client.post("/models/validate", json={"model": bad})
        assert r.json()["valid"] is False
        assert r.json()["error"]["category"] == "input_error"

        payload = {
            "sut": "reference", "model": model, "seed": 11,
            "fuzz_steps": 8, "db_path": db, "replay_dir": replays,
        }
        r = await client.post("/runs", json=payload)
        assert r.status_code == 200, r.text
        ref = r.json()
        assert ref["summary"]["findings"] == 0

        payload["sut"] = "buggy"
        r = await client.post("/runs", json=payload)
        buggy = r.json()
        assert buggy["summary"]["findings"] == 3
        assert buggy["summary"]["finding_categories"] == ["state_conflict"]

        run_id = buggy["run_id"]
        r = await client.get(f"/runs/{run_id}", params={"db_path": db})
        assert r.json()["status"] == "completed"
        assert r.json()["config"]["seed"] == 11

        r = await client.get(f"/runs/{run_id}/steps", params={"db_path": db})
        steps = r.json()["steps"]
        assert len(steps) == buggy["summary"]["steps_executed"]
        first = steps[0]
        assert first["decision"]["mode"] == "guided_repeated_close"
        assert "rng_state_before" in first["decision"]
        assert first["model_state_before"] == "draft"
        assert first["expected"]["reason"]

        r = await client.get(f"/runs/{run_id}/findings", params={"db_path": db})
        findings = r.json()["findings"]
        assert len(findings) == 3
        assert "replay_script" not in findings[0]

        r = await client.get(
            f"/runs/{run_id}/findings/0/replay", params={"db_path": db}
        )
        assert r.status_code == 200
        assert "SEQUENCE" in r.text and "urllib.request" in r.text

        r = await client.get("/runs", params={"db_path": db})
        assert len(r.json()["runs"]) == 2
        r = await client.get("/runs/does-not-exist", params={"db_path": db})
        assert r.status_code == 404

        r = await client.post(
            "/runs",
            json={"sut": "http", "model": model, "db_path": db,
                  "replay_dir": replays},
        )
        assert r.status_code == 422
        assert r.json()["detail"]["category"] == "input_error"


def test_api_contracts(paths):
    asyncio.run(_scenario(*paths))
