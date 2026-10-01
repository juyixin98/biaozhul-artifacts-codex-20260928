#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m compileall -q app experiments
python3 experiments/generate_fixtures.py
python3 experiments/reproduce.py
pytest -q
