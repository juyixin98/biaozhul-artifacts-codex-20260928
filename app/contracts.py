"""Request/response contracts: the sample-contract layer of the service."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Observations must be finite; NaN/inf are rejected at the boundary.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

_MAX_GROUP_LEN = 100_000


class PermutationTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_a: list[FiniteFloat] = Field(min_length=1, max_length=_MAX_GROUP_LEN)
    group_b: list[FiniteFloat] = Field(min_length=1, max_length=_MAX_GROUP_LEN)
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    # Monte Carlo sample size; ignored when the exact path is taken.
    n_permutations: int | None = Field(default=None, ge=10, le=2_000_000)
    # Seed for the Monte Carlo path; generated and reported when omitted.
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    # Per-request override of the exact/Monte-Carlo switch-over point.
    exact_threshold: int | None = Field(default=None, ge=1, le=1_000_000)
    # Hard budget on evaluated permutations; hitting it cancels the run.
    max_evaluations: int | None = Field(default=None, ge=1)
    # Attach the full null distribution (exact path, capped sizes only).
    include_distribution: bool = False


class PermutationTestResponse(BaseModel):
    run_id: str
    status: Literal["completed", "cancelled"]
    method: Literal["exact", "monte_carlo"] | None
    statistic: float
    p_value: float | None
    # Number of valid permutations actually evaluated.
    n_permutations: int
    # Total size of the permutation space C(n_a + n_b, n_a).
    n_combinations: int
    mc_std_error: float | None
    alternative: str
    seed: int | None
    group_sizes: list[int]
    input_hash: str
    kernel_version: str
    library_versions: dict[str, str]
    created_at: str
    distribution: list[float] | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody
