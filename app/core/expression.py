from __future__ import annotations

import ast
import math
from collections.abc import Callable

import numpy as np

from app.core import dual
from app.core.diagnostics import DerivativeCheck, FitError, FitStatus


ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


class RestrictedExpression:
    """A one-dimensional model f(x, theta) built from an allow-listed AST."""

    def __init__(
        self,
        source: str,
        parameter_names: list[str],
        *,
        variable_name: str = "x",
    ):
        self.source = source
        self.parameter_names = list(parameter_names)
        self.variable_name = variable_name
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise FitError(
                FitStatus.INVALID_EXPRESSION,
                f"expression is not valid Python: {exc.msg}",
            ) from exc
        self._validate(tree)
        self.tree = tree

    def _validate(self, tree: ast.AST) -> None:
        allowed = ALLOWED_NODES
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise FitError(
                    FitStatus.INVALID_EXPRESSION,
                    f"forbidden expression construct: {type(node).__name__}",
                )
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.keywords:
                    raise FitError(
                        FitStatus.INVALID_EXPRESSION,
                        "only simple allow-listed unary functions are supported",
                    )
                if len(node.args) != 1:
                    raise FitError(
                        FitStatus.INVALID_EXPRESSION,
                        f"{node.func.id} expects exactly one argument",
                    )
                if node.func.id not in dual.FUNCTIONS:
                    raise FitError(
                        FitStatus.INVALID_EXPRESSION,
                        f"function is not allowed: {node.func.id}",
                    )
            if isinstance(node, ast.Name) and node.id not in (
                self.variable_name,
                *self.parameter_names,
                *dual.FUNCTIONS.keys(),
            ):
                raise FitError(
                    FitStatus.INVALID_EXPRESSION,
                    f"unknown identifier: {node.id}",
                )
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise FitError(
                    FitStatus.INVALID_EXPRESSION,
                    "only numeric constants are supported",
                )

    def _eval_node(
        self,
        node: ast.AST,
        values: dict[str, dual.Dual | float],
    ) -> dual.Dual | float:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, values)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.UnaryOp):
            value = self._eval_node(node.operand, values)
            if isinstance(node.op, ast.USub):
                return dual.neg(value)
            if isinstance(node.op, ast.UAdd):
                return dual.pos(value)
            raise FitError(FitStatus.INVALID_EXPRESSION, "unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, values)
            right = self._eval_node(node.right, values)
            if isinstance(node.op, ast.Add):
                return dual.add(left, right)
            if isinstance(node.op, ast.Sub):
                return dual.sub(left, right)
            if isinstance(node.op, ast.Mult):
                return dual.mul(left, right)
            if isinstance(node.op, ast.Div):
                return dual.div(left, right)
            if isinstance(node.op, ast.Pow):
                return dual.power(left, right)
            raise FitError(FitStatus.INVALID_EXPRESSION, "unsupported binary operator")
        if isinstance(node, ast.Call):
            return dual.call_function(
                node.func.id, self._eval_node(node.args[0], values)
            )
        raise FitError(FitStatus.INVALID_EXPRESSION, "unsupported expression node")

    def evaluate(self, x: float, theta: np.ndarray) -> float:
        values: dict[str, dual.Dual | float] = {
            self.variable_name: float(x),
        }
        for i, name in enumerate(self.parameter_names):
            values[name] = float(theta[i])
        value = self._eval_node(self.tree, values)
        result = dual._real(value)
        if not np.isfinite(result):
            raise ValueError("model returned a non-finite value")
        return result

    def evaluate_with_jacobian(self, x: float, theta: np.ndarray) -> tuple[float, np.ndarray]:
        values: dict[str, dual.Dual | float] = {
            self.variable_name: float(x),
        }
        for i, name in enumerate(self.parameter_names):
            values[name] = dual.Dual(float(theta[i]), np.eye(len(theta))[i])
        result = self._eval_node(self.tree, values)
        if not isinstance(result, dual.Dual):
            return float(result), np.zeros(len(theta), dtype=float)
        if not np.isfinite(result.real) or not np.all(np.isfinite(result.grad)):
            raise ValueError("model or derivative returned a non-finite value")
        return float(result.real), np.asarray(result.grad, dtype=float)


class ExpressionModel:
    def __init__(
        self,
        expression: RestrictedExpression,
        x_data: np.ndarray,
        y_data: np.ndarray,
        weights: np.ndarray | None = None,
    ):
        self.expression = expression
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        self.weights = (
        np.ones_like(self.y_data, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
        )
        if self.x_data.shape != self.y_data.shape:
            raise FitError(FitStatus.INVALID_INPUT, "x and y must have equal length")
        if self.weights.shape != self.y_data.shape:
            raise FitError(FitStatus.INVALID_INPUT, "weights and y must have equal length")

    @property
    def parameter_names(self) -> list[str]:
        return self.expression.parameter_names

    def residual(self, theta: np.ndarray) -> np.ndarray:
        try:
            values = np.array(
                [self.expression.evaluate(x, theta) for x in self.x_data],
                dtype=float,
            )
            return self.weights * (values - self.y_data)
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FitError(
                FitStatus.INVALID_INITIAL_RESIDUAL,
                f"model value is not finite: {exc}",
            ) from exc

    def residual_and_jacobian(
        self,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            evaluated = [
                self.expression.evaluate_with_jacobian(x, theta) for x in self.x_data
            ]
            values = np.array([item[0] for item in evaluated], dtype=float)
            jacobian = np.array([item[1] for item in evaluated], dtype=float)
            return self.weights * (values - self.y_data), self.weights[:, None] * jacobian
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FitError(
                FitStatus.INVALID_INITIAL_RESIDUAL,
                f"model value or derivative is not finite: {exc}",
            ) from exc


def finite_difference_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    relative_step: float = 1e-7,
) -> np.ndarray:
    lower, upper = bounds
    n = theta.size
    base = residual_fn(theta)
    jacobian = np.zeros((base.size, n), dtype=float)
    for i in range(n):
        h = max(float(relative_step), math.sqrt(np.finfo(float).eps))
        h *= max(1.0, abs(float(theta[i])), abs(float(lower[i])), abs(float(upper[i])))
        forward = theta[i] + h <= upper[i]
        backward = theta[i] - h >= lower[i]
        plus = theta.copy()
        minus = theta.copy()
        if forward and backward:
            plus[i] += h
            minus[i] -= h
            numerical = (residual_fn(plus) - residual_fn(minus)) / (2.0 * h)
        elif forward:
            plus[i] += h
            numerical = (residual_fn(plus) - base) / h
        elif backward:
            minus[i] -= h
            numerical = (base - residual_fn(minus)) / h
        else:
            raise FitError(
                FitStatus.DERIVATIVE_MISMATCH,
                f"parameter {i} has no finite-difference room inside its bounds",
            )
        jacobian[:, i] = numerical
    return jacobian


def check_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    analytical_jacobian: np.ndarray,
    theta: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    parameter_names: list[str],
    *,
    tolerance: float = 5e-5,
    relative_step: float = 1e-7,
) -> DerivativeCheck:
    numerical = finite_difference_jacobian(
        residual_fn,
        theta,
        bounds,
        relative_step=relative_step,
    )
    if numerical.shape != analytical_jacobian.shape or not np.all(
        np.isfinite(numerical)
    ):
        raise FitError(
            FitStatus.DERIVATIVE_MISMATCH,
            "finite-difference Jacobian had an invalid shape or non-finite entries",
        )
    scale = np.maximum(
        np.maximum(1.0, np.abs(analytical_jacobian)),
        np.abs(numerical),
    )
    errors = np.abs(analytical_jacobian - numerical) / scale
    if not np.all(np.isfinite(errors)):
        raise FitError(
            FitStatus.DERIVATIVE_MISMATCH,
            "analytical and finite-difference Jacobians are not both finite",
        )
    flat_index = int(np.argmax(errors))
    observation, parameter = np.unravel_index(flat_index, errors.shape)
    return DerivativeCheck(
        passed=bool(np.max(errors) <= tolerance),
        max_relative_error=float(np.max(errors)),
        worst_parameter=parameter_names[parameter],
        worst_observation=int(observation),
        relative_step=float(relative_step),
    )
