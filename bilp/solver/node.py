"""Search-tree nodes: branch constraints, bound materialization, restoration.

A node is the root problem plus a path of branch constraints
(``x[j] <= k`` / ``x[j] >= k``). Bounds are *materialized* by folding the
branch constraints into the base variable bounds. The archive keeps every
created node so any node can be restored later: restoration recomputes the
bounds from the stored branch constraints and verifies them against the
materialized copy recorded when the node was processed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import ComputationFailureError, RunNotFoundError


@dataclass(frozen=True)
class BranchConstraint:
    var: int
    sense: str   # "<=" or ">="
    value: float  # integral bound produced by floor/ceil of the parent LP point

    def to_dict(self) -> dict:
        return {"var": self.var, "sense": self.sense, "value": float(self.value)}


@dataclass(frozen=True)
class Node:
    node_id: int
    parent_id: int | None
    depth: int
    constraints: tuple[BranchConstraint, ...]  # full path from the root


def materialize_bounds(base_lb: np.ndarray, base_ub: np.ndarray,
                       constraints: tuple[BranchConstraint, ...]) -> tuple[np.ndarray, np.ndarray]:
    lb = base_lb.copy()
    ub = base_ub.copy()
    for bc in constraints:
        if bc.sense == "<=":
            ub[bc.var] = min(ub[bc.var], bc.value)
        else:
            lb[bc.var] = max(lb[bc.var], bc.value)
    return lb, ub


@dataclass
class NodeArchive:
    """Registry of every created node for later restoration."""

    base_lb: np.ndarray
    base_ub: np.ndarray
    bounds_retention: int = 1000  # materialized bounds kept for the first N nodes
    _nodes: dict[int, Node] = field(default_factory=dict)
    _bounds: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def register(self, node: Node) -> tuple[np.ndarray, np.ndarray]:
        lb, ub = materialize_bounds(self.base_lb, self.base_ub, node.constraints)
        self._nodes[node.node_id] = node
        if node.node_id not in self._bounds and len(self._bounds) < self.bounds_retention:
            self._bounds[node.node_id] = (lb.copy(), ub.copy())
        return lb, ub

    def restore(self, node_id: int) -> dict:
        """Restore a node: branch constraints + bounds, recomputed from the path."""
        node = self._nodes.get(node_id)
        if node is None:
            raise RunNotFoundError(
                f"node {node_id} is not in the archive",
                {"node_id": node_id},
            )
        lb, ub = materialize_bounds(self.base_lb, self.base_ub, node.constraints)
        recorded = self._bounds.get(node_id)
        if recorded is not None:
            if not (np.array_equal(lb, recorded[0]) and np.array_equal(ub, recorded[1])):
                raise ComputationFailureError(
                    f"restored bounds for node {node_id} do not match the recorded "
                    "materialized bounds (archive inconsistency)",
                    {"node_id": node_id},
                )
        return {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "branch_constraints": [bc.to_dict() for bc in node.constraints],
            "lb": [None if np.isinf(v) and v < 0 else float(v) for v in lb],
            "ub": [None if np.isinf(v) and v > 0 else float(v) for v in ub],
        }
