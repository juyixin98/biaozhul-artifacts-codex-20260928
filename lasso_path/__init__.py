"""Lasso regularization-path backend.

Module responsibilities
-----------------------
config         Frozen runtime settings (env-overridable via ``LASSO_`` prefix).
diagnostics    Structured accept/reject/undetermined records carrying request ids.
models         Pydantic request/response schemas for the service layer.
preprocessing  Frozen centering/scaling statistics and scale restoration.
solver         Coordinate-descent core (soft threshold, duality gap / KKT).
validation     Input validation (shapes, finiteness, zero-variance columns).
path           Regularization-path orchestration with warm starts.
service        FastAPI entry point.
"""

__version__ = "0.1.0"
