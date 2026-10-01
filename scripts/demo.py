from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def main() -> None:
    rng = np.random.default_rng(20260930)
    x = np.linspace(0.0, 3.0, 32)
    true = {"a": 2.8, "b": 0.55, "c": 0.35}
    y = true["a"] * np.exp(-true["b"] * x) + true["c"]
    y += rng.normal(scale=1.0e-6, size=x.size)

    payload = {
        "expression": "a*exp(-b*x)+c",
        "parameter_names": ["a", "b", "c"],
        "x": x.tolist(),
        "y": y.tolist(),
        "initial": [0.1, 3.0, 0.001],
        "bounds": {
            "lower": [0.0, 0.0, 0.0],
            "upper": [10.0, 5.0, 10.0],
        },
        "scales": [3.0, 1.0, 0.5],
        "options": {
            "check_derivatives": True,
            "max_iterations": 100,
        },
    }

    client = TestClient(app)
    response = client.post("/api/v1/least-squares/fit", json=payload)
    print(f"HTTP {response.status_code}")
    response.raise_for_status()
    result = response.json()

    print("Independent synthetic reference:", json.dumps(true, sort_keys=True))
    print("run_id:", result["run_id"])
    print("input_fingerprint:", result["input_fingerprint"])
    print("status:", result["status"], "success:", result["success"])
    print("message:", result["message"])
    print("reason:", result["reason"])
    print("parameters:", json.dumps(result["parameters"], sort_keys=True))
    print("residual_norm:", float(np.linalg.norm(result["residuals"])))
    print("cost:", result["cost"])
    print("stationarity_type:", result["stationarity_type"])
    print("identifiability:", json.dumps(result["identifiability"], sort_keys=True))
    print(
        "iter actual_reduction predicted_reduction ratio accepted "
        "active_lower active_upper bound_hit"
    )
    for item in result["iterations"]:
        print(
            item["iteration"],
            item["actual_reduction"],
            item["predicted_reduction"],
            item["reduction_ratio"],
            item["accepted"],
            ",".join(item["active_lower"]) or "-",
            ",".join(item["active_upper"]) or "-",
            ",".join(item["bound_hit"]) or "-",
            item["judgment"],
        )


if __name__ == "__main__":
    main()
