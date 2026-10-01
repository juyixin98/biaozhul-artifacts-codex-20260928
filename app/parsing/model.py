"""Restricted expression parser and forward-mode automatic differentiation.

The grammar intentionally accepts only mathematical expressions.  Python
attribute access, calls to arbitrary functions, names unrelated to ``x`` and
declared parameters, and control-flow nodes are rejected before evaluation.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.status import SolverError, Status


class InvalidExpression(SolverError):
    def __init__(self, message: str) -> None:
        super().__init__(
            Status.INVALID_EXPRESSION,
            message,
            http_status_code=400,
        )


@dataclass(frozen=True)
class Dual:
    value: Any
    grad: Any = None


def _constant(value: Any, n: int) -> Dual:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(n, float(array))
    return Dual(array, np.zeros_like(array))


def _wrap(value: Any, like: Dual) -> Dual:
    if isinstance(value, Dual):
        return value
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        grad = None
    else:
        grad = np.zeros_like(array)
    return Dual(array, grad)


def _binop(op: ast.AST, left: Dual, right: Dual) -> Dual:
    with np.errstate(all="ignore"):
        a, b = left.value, right.value
        ga, gb = left.grad, right.grad

        if isinstance(op, ast.Add):
            value = a + b
            grad = None if ga is None and gb is None else _zero_add(ga, a, gb, b)
        elif isinstance(op, ast.Sub):
            value = a - b
            if ga is None and gb is None:
                grad = None
            elif ga is None:
                grad = -_broadcast_grad(gb, value)
            elif gb is None:
                grad = _broadcast_grad(ga, value)
            else:
                grad = _broadcast_grad(ga, value) - _broadcast_grad(gb, value)
        elif isinstance(op, ast.Mult):
            value = a * b
            grad = None
            if ga is not None:
                grad = _broadcast_grad(ga, value) * b
            if gb is not None:
                term = _broadcast_grad(gb, value) * a
                grad = term if grad is None else grad + term
        elif isinstance(op, ast.Div):
            value = a / b
            grad = None
            if ga is not None:
                grad = _broadcast_grad(ga, value) / b
            if gb is not None:
                term = _broadcast_grad(gb, value) * a / (b * b)
                grad = -term if grad is None else grad - term
        elif isinstance(op, ast.Pow):
            value = np.power(a, b)
            grad = None
            if ga is not None:
                grad = _broadcast_grad(ga, value) * b * np.power(a, b - 1.0)
            if gb is not None:
                # d(a^b)/db = a^b log(a); invalid domains become NaN and are
                # reported explicitly by the solver's finite-value checks.
                term = _broadcast_grad(gb, value) * value * np.log(a)
                grad = term if grad is None else grad + term
        else:
            raise InvalidExpression(f"unsupported binary operator: {type(op).__name__}")
        return Dual(value, grad)


def _zero_add(ga: Any, a: Any, gb: Any, b: Any) -> np.ndarray:
    value = np.asarray(a) + np.asarray(b)
    left = np.zeros_like(value) if ga is None else _broadcast_grad(ga, value)
    right = np.zeros_like(value) if gb is None else _broadcast_grad(gb, value)
    return left + right


def _broadcast_grad(grad: Any, value: Any) -> np.ndarray:
    grad_array = np.asarray(grad, dtype=float)
    value_array = np.asarray(value, dtype=float)
    return np.broadcast_to(grad_array, value_array.shape).copy()


FUNCTIONS: dict[str, Callable[[Dual], Dual]] = {}


def _register(name: str, value_fn: Callable[[Any], Any], derivative_fn: Callable[[Any], Any]) -> None:
    def derivative(argument: Dual) -> Dual:
        with np.errstate(all="ignore"):
            value = value_fn(argument.value)
        grad = None
        if argument.grad is not None:
            with np.errstate(all="ignore"):
                grad = _broadcast_grad(argument.grad, value) * derivative_fn(argument.value)
        return Dual(value, grad)

    FUNCTIONS[name] = derivative


_register("exp", np.exp, np.exp)
_register("expm1", np.expm1, np.exp)
_register("log", np.log, lambda x: 1.0 / x)
_register("log1p", np.log1p, lambda x: 1.0 / (1.0 + x))
_register("sqrt", np.sqrt, lambda x: 0.5 / np.sqrt(x))
_register("sin", np.sin, np.cos)
_register("cos", np.cos, lambda x: -np.sin(x))
_register("tan", np.tan, lambda x: 1.0 / np.square(np.cos(x)))
_register("atan", np.arctan, lambda x: 1.0 / (1.0 + np.square(x)))
_register("arctan", np.arctan, lambda x: 1.0 / (1.0 + np.square(x)))

CONSTANTS = {"pi": math.pi, "e": math.e}


class CompiledModel:
    """A validated scalar mathematical model ``y = f(x, parameters)``."""

    def __init__(self, expression: str, parameter_names: list[str], n_points: int) -> None:
        if not expression or not expression.strip():
            raise InvalidExpression("model expression must not be empty")
        if len(parameter_names) != len(set(parameter_names)):
            raise InvalidExpression("parameter names must be unique")
        if any(name in {"x", "pi", "e"} for name in parameter_names):
            raise InvalidExpression("parameter names cannot be x, pi, or e")
        if n_points <= 0:
            raise SolverError(Status.INVALID_REQUEST, "x and y must contain data", http_status_code=400)

        self.expression = expression
        self.parameter_names = list(parameter_names)
        self.n_points = n_points
        try:
            self.tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise InvalidExpression(f"syntax error: {exc.msg}") from exc
        self._validate_node(self.tree.body, set(self.parameter_names))

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    def _validate_node(self, node: ast.AST, parameters: set[str]) -> None:
        if isinstance(node, ast.Expression):
            self._validate_node(node.body, parameters)
        elif isinstance(node, ast.BinOp):
            if not isinstance(
                node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow),
            ):
                raise InvalidExpression(f"unsupported operator: {type(node.op).__name__}")
            self._validate_node(node.left, parameters)
            self._validate_node(node.right, parameters)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise InvalidExpression(f"unsupported unary operator: {type(node.op).__name__}")
            self._validate_node(node.operand, parameters)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise InvalidExpression("only simple allowed mathematical functions may be called")
            if node.func.id not in FUNCTIONS:
                raise InvalidExpression(f"function is not allowed: {node.func.id}")
            if node.keywords or len(node.args) != 1:
                raise InvalidExpression(f"{node.func.id} must be called with one positional argument")
            self._validate_node(node.args[0], parameters)
        elif isinstance(node, ast.Name):
            if node.id != "x" and node.id not in parameters and node.id not in CONSTANTS:
                raise InvalidExpression(f"unknown identifier: {node.id}")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise InvalidExpression("only numeric constants are allowed")
            if not math.isfinite(float(node.value)):
                raise InvalidExpression("numeric constants must be finite")
        else:
            raise InvalidExpression(f"expression node is not allowed: {type(node).__name__}")

    def evaluate(self, parameter_values: np.ndarray, x: np.ndarray) -> np.ndarray:
        result = self._evaluate(parameter_values, x, None)
        value = np.asarray(result.value, dtype=float)
        return np.broadcast_to(value, (self.n_points,)).copy()

    def jacobian(self, parameter_values: np.ndarray, x: np.ndarray) -> np.ndarray:
        columns = []
        for index in range(self.n_parameters):
            result = self._evaluate(parameter_values, x, index)
            value = np.asarray(result.value, dtype=float)
            if result.grad is None:
                column = np.zeros(self.n_points)
            else:
                column = np.broadcast_to(np.asarray(result.grad, dtype=float), value.shape).reshape(-1)
                column = np.broadcast_to(column, (self.n_points,)).copy()
            columns.append(column)
        return np.column_stack(columns) if columns else np.empty((self.n_points, 0))

    def _evaluate(self, parameter_values: np.ndarray, x: np.ndarray, seed_index: int | None) -> Dual:
        env: dict[str, Dual | float] = {"x": Dual(np.asarray(x, dtype=float), np.zeros(self.n_points))}
        for index, name in enumerate(self.parameter_names):
            value = float(parameter_values[index])
            grad = 1.0 if index == seed_index else 0.0
            env[name] = Dual(value, grad)
        for name, value in CONSTANTS.items():
            env[name] = Dual(value, None)
        return self._eval_node(self.tree.body, env)

    def _eval_node(self, node: ast.AST, env: dict[str, Dual | float]) -> Dual:
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, env)
            right = self._eval_node(node.right, env)
            return _binop(node.op, left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, env)
            if isinstance(node.op, ast.UAdd):
                return operand
            return Dual(-operand.value, None if operand.grad is None else -operand.grad)
        if isinstance(node, ast.Call):
            return FUNCTIONS[node.func.id](self._eval_node(node.args[0], env))
        if isinstance(node, ast.Name):
            value = env[node.id]
            return value if isinstance(value, Dual) else Dual(float(value), None)
        if isinstance(node, ast.Constant):
            return Dual(float(node.value), None)
        raise InvalidExpression(f"expression node is not allowed: {type(node).__name__}")
