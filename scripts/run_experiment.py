from __future__ import annotations

import argparse
import json

from survey_imputation.config import get_settings
from survey_imputation.experiment import run_comparison, write_evidence
from survey_imputation.repository import SQLiteResultRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic missingness comparison")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--missing-probability", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    settings = get_settings()
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    repository = SQLiteResultRepository(settings.database_path)
    rows = run_comparison(
        repository=repository,
        n=args.n,
        missing_probability=args.missing_probability,
        m=args.m,
        base_seed=args.seed,
    )
    output_path = settings.results_dir / f"comparison_seed_{args.seed}_n_{args.n}.json"
    write_evidence(rows, output_path)
    print(json.dumps({"rows": len(rows), "output": str(output_path), "database": str(settings.database_path)}))


if __name__ == "__main__":
    main()
