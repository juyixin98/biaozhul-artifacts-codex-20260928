"""bilp: bounded integer linear programming backend.

Branch-and-bound over LP relaxations solved by SciPy/HiGHS.
Modules:
  - model:    request parsing and validation into a canonical problem
  - solver:   branch-and-bound kernel (nodes, evidence, numerics)
  - config:   standalone solver configuration
  - errors:   error taxonomy (input / conflict / resource / computation)
  - runs:     run registry with replay support
  - main:     FastAPI service entrypoint
"""

__version__ = "0.1.0"
