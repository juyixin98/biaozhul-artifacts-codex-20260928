#!/usr/bin/env python3
"""Run the full local test suite with pinned dependencies."""

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"])


if __name__ == "__main__":
    raise SystemExit(main())
