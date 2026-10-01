from dataclasses import dataclass
import os


SERVICE_NAME = "constrained-qp-active-set"
SOLVER_NAME = "primal-active-set-qp"
SERVICE_VERSION = "1.0.0"


@dataclass(frozen=True)
class Tolerances:
    primal: float = 1e-8
    dual: float = 1e-8
    complementarity: float = 1e-8
    active: float = 1e-9
    rank: float = 1e-11
    step: float = 1e-10
    symmetry: float = 1e-9
    positive_definite: float = 1e-10

    @classmethod
    def from_api(cls, data: dict | None) -> "Tolerances":
        allowed = {
            "primal_tol": "primal",
            "dual_tol": "dual",
            "complementarity_tol": "complementarity",
            "active_tol": "active",
            "rank_tol": "rank",
            "step_tol": "step",
            "symmetry_tol": "symmetry",
            "positive_definite_tol": "positive_definite",
        }
        values = {}
        for api_name, field_name in allowed.items():
            if data is not None and data.get(api_name) is not None:
                values[field_name] = float(data[api_name])
        return cls(**values)


def log_level() -> str:
    return os.getenv("QP_LOG_LEVEL", "INFO").upper()
