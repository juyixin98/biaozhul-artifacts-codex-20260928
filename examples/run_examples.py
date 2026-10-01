import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models import FeasibilityRequest, SolveRequest
from app.service import find_feasibility, solve_qp


EXAMPLES_DIR = Path(__file__).resolve().parent


def main() -> None:
    for name in ("boundary_qp.json", "infeasible_qp.json"):
        payload = json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))
        response = solve_qp(SolveRequest.model_validate(payload))
        print(f"\n=== {name} ===")
        print(json.dumps(response, indent=2, ensure_ascii=False))

    payload = json.loads((EXAMPLES_DIR / "feasibility_only.json").read_text(encoding="utf-8"))
    response = find_feasibility(FeasibilityRequest.model_validate(payload))
    print("\n=== feasibility_only.json ===")
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
