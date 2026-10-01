"""Restricted expression parsing and reverse-mode automatic differentiation."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class ExpressionValidationError(ValueError):
    """The expression is syntactically or semantically outside the DSL."""


class IllegalFunctionError(ValueError):
    """Evaluation produced a non-finite value outside the function domain."""


@dataclass(frozen=True)
class Node:
    tape: list["Node"]
    index: int
    value: np.ndarray
    kind: str
    inputs: tuple[int, ...] = ()
    scalar_broadcast: bool = False

    def as_node(self, other: Any) -> "Node":
        if isinstance(other, Node):
            return other
        if isinstance(other, (int, float, np.number, np.ndarray)):
            node = Node(self.tape, len(self.tape), np.asarray(other, dtype=float), "constant")
            self.tape.append(node)
            return node
        raise ExpressionValidationError(f"Unsupported operand type: {type(other)!r}")

    def __add__(self, other: Any) -> "Node":
        b = self.as_node(other)
        value = self.value + b.value
        scalar_broadcast = value.shape == () and (self.scalar_broadcast or b.scalar_broadcast)
        node = Node(self.tape, len(self.tape), value, "add", (self.index, b.index), scalar_broadcast)
        self.tape.append(node)
        return node

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Node":
        b = self.as_node(other)
        value = self.value - b.value
        scalar_broadcast = value.shape == () and (self.scalar_broadcast or b.scalar_broadcast)
        node = Node(self.tape, len(self.tape), value, "sub", (self.index, b.index), scalar_broadcast)
        self.tape.append(node)
        return node

    def __rsub__(self, other: Any) -> "Node":
        b = self.as_node(other)
        value = b.value - self.value
        scalar_broadcast = value.shape == () and (self.scalar_broadcast or b.scalar_broadcast)
        node = Node(self.tape, len(self.tape), value, "sub", (b.index, self.index), scalar_broadcast)
        self.tape.append(node)
        return node

    def __mul__(self, other: Any) -> "Node":
        b = self.as_node(other)
        value = self.value * b.value
        scalar_broadcast = value.shape == () and (self.scalar_broadcast or b.scalar_broadcast)
        node = Node(self.tape, len(self.tape), value, "mul", (self.index, b.index), scalar_broadcast)
        self.tape.append(node)
        return node

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "Node":
        b = self.as_node(other)
        value = self.value / b.value
        scalar_broadcast = value.shape == () and (self.scalar_broadcast or b.scalar_broadcast)
        node = Node(self.tape, len(self.tape), value, "div", (self.index, b.index), scalar_broadcast)
        self.tape.append(node)
        return node

    def __rtruediv__(self, other: Any) -> "Node":
        b = self.as_node(other)
        value = b.value / self.value
        scalar_broadcast = value.shape == () and (self.scalar_broadcast or b.scalar_broadcast)
        node = Node(self.tape, len(self.tape), value, "div", (b.index, self.index), scalar_broadcast)
        self.tape.append(node)
        return node

    def __pow__(self, other: Any) -> "Node":
        b = self.as_node(other)
        value = self.value**b.value
        scalar_broadcast = value.shape == () and (self.scalar_broadcast or b.scalar_broadcast)
        node = Node(self.tape, len(self.tape), value, "pow", (self.index, b.index), scalar_broadcast)
        self.tape.append(node)
        return node

    def __rpow__(self, other: Any) -> "Node":
        b = self.as_node(other)
        value = b.value**self.value
        scalar_broadcast = value.shape == () and (self.scalar_broadcast or b.scalar_broadcast)
        node = Node(self.tape, len(self.tape), value, "pow", (b.index, self.index), scalar_broadcast)
        self.tape.append(node)
        return node

    def __neg__(self) -> "Node":
        node = Node(self.tape, len(self.tape), -self.value, "neg", (self.index,), self.scalar_broadcast)
        self.tape.append(node)
        return node

    def __pos__(self) -> "Node":
        return self


ALLOWED_FUNCTIONS = frozenset({"exp", "log", "sqrt", "sin", "cos", "tan", "log10", "abs"})
ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}


def unary_value(name: str, value: np.ndarray) -> np.ndarray:
    table = {
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "log10": np.log10,
        "abs": abs,
    }
    return table[name](value)


class ExpressionModel:
    """A compiled model expression evaluated at scalar parameters and vector x."""

    def __init__(self, expression: str, parameter_names: list[str]):
        self.expression = expression
        self.parameter_names = list(parameter_names)
        if not self.parameter_names:
            raise ExpressionValidationError("At least one parameter is required")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ExpressionValidationError("Parameter names must be unique")
        if "x" in self.parameter_names:
            raise ExpressionValidationError("'x' is reserved for independent data")
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ExpressionValidationError(f"Invalid expression syntax: {exc.msg}") from exc
        self._tree = parsed.body
        self._validate_tree(self._tree)

    def _validate_tree(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Constant,
            ast.Load,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
        )
        if not isinstance(node, allowed):
            raise ExpressionValidationError(f"Unsupported syntax: {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            self._validate_tree(child)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
                raise ExpressionValidationError("Only allowlisted mathematical functions may be called")
            if len(node.args) != 1 or node.keywords:
                raise ExpressionValidationError("Functions accept exactly one positional argument")
        allowed_names = [*self.parameter_names, "x", *ALLOWED_CONSTANTS, *ALLOWED_FUNCTIONS]
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ExpressionValidationError(f"Unknown name: {node.id}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ExpressionValidationError("Only numeric constants are supported")

    def evaluate(self, x: np.ndarray, parameters: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Return model values and dense Jacobian with respect to parameters."""
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim != 1:
            raise ExpressionValidationError("Independent data x must be one-dimensional")
        tape: list[Node] = []
        leaf_ids: dict[str, int] = {}
        environment: dict[str, Any] = {name: ALLOWED_CONSTANTS[name] for name in ALLOWED_CONSTANTS}

        x_node = Node(tape, len(tape), x_array, "leaf")
        tape.append(x_node)
        leaf_ids["x"] = x_node.index
        environment["x"] = x_node

        for name in self.parameter_names:
            try:
                value = float(parameters[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ExpressionValidationError(f"Missing or non-numeric parameter: {name}") from exc
            if not math.isfinite(value):
                raise IllegalFunctionError(f"Parameter {name} is not finite")
            node = Node(tape, len(tape), np.asarray(value, dtype=float), "leaf", scalar_broadcast=True)
            tape.append(node)
            leaf_ids[name] = node.index
            environment[name] = node

        for name in ALLOWED_FUNCTIONS:
            environment[name] = lambda arg, func_name=name: self._call_function(tape, func_name, arg)

        try:
            with np.errstate(all="ignore"):
                output = self._evaluate_node(self._tree, tape, environment)
                values = np.asarray(output.value, dtype=float)
                if values.shape != x_array.shape:
                    raise ExpressionValidationError(
                        f"Model output shape {values.shape} does not match data shape {x_array.shape}"
                    )
                grads = self._backward(tape, output)
                jacobian = np.column_stack([grads[leaf_ids[name]] for name in self.parameter_names])
        except ExpressionValidationError:
            raise
        except IllegalFunctionError:
            raise
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, FloatingPointError) as exc:
            raise IllegalFunctionError(str(exc)) from exc

        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(jacobian)):
            raise IllegalFunctionError("Model value or derivative contains NaN or infinity")
        return values, jacobian

    def _call_function(self, tape: list[Node], name: str, arg: Any) -> Node:
        if not isinstance(arg, Node):
            arg = Node(tape, len(tape), np.asarray(float(arg), dtype=float), "constant")
            tape.append(arg)
        node = Node(
            tape,
            len(tape),
            unary_value(name, arg.value),
            name,
            (arg.index,),
            arg.scalar_broadcast,
        )
        tape.append(node)
        return node

    def _evaluate_node(self, node: ast.AST, tape: list[Node], env: Mapping[str, Any]) -> Node:
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Constant):
            n = Node(tape, len(tape), np.asarray(float(node.value), dtype=float), "constant")
            tape.append(n)
            return n
        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate_node(node.operand, tape, env)
            return -operand if isinstance(node.op, ast.USub) else +operand
        if isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left, tape, env)
            right = self._evaluate_node(node.right, tape, env)
            operations = {
                ast.Add: left.__add__,
                ast.Sub: left.__sub__,
                ast.Mult: left.__mul__,
                ast.Div: left.__truediv__,
                ast.Pow: left.__pow__,
            }
            return operations[type(node.op)](right)
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            arg = self._evaluate_node(node.args[0], tape, env)
            return self._call_function(tape, node.func.id, arg)
        raise ExpressionValidationError(f"Unsupported syntax: {type(node).__name__}")

    def _backward(self, tape: list[Node], output: Node) -> dict[int, np.ndarray]:
        gradients = {output.index: np.ones_like(output.value)}

        def unbroadcast(value: np.ndarray, index: int) -> np.ndarray:
            target = tape[index].value
            if target.shape == ():
                if tape[index].scalar_broadcast:
                    return np.asarray(value, dtype=float)
                return np.asarray(np.sum(value), dtype=float)
            reduced = np.broadcast_to(value, node.value.shape).copy() if value.shape != node.value.shape else value
            shape = target.shape
            while reduced.ndim > len(shape):
                reduced = reduced.sum(axis=0)
            for axis, size in enumerate(shape):
                if reduced.ndim > axis and size == 1 and reduced.shape[axis] != 1:
                    reduced = reduced.sum(axis=axis, keepdims=True)
            return np.reshape(reduced, shape)

        def add_grad(index: int, value: np.ndarray) -> None:
            value = unbroadcast(value, index)
            gradients[index] = value + gradients[index] if index in gradients else value

        for node in reversed(tape):
            grad = gradients.get(node.index)
            if grad is None or node.kind in {"leaf", "constant"}:
                continue
            i = node.inputs
            if node.kind == "neg":
                add_grad(i[0], -grad)
            elif node.kind == "abs":
                add_grad(i[0], grad * np.sign(tape[i[0]].value))
            elif node.kind in {"exp", "log", "sqrt", "sin", "cos", "tan", "log10"}:
                a = tape[i[0]].value
                if node.kind == "exp":
                    local = np.exp(a)
                elif node.kind == "log":
                    local = 1.0 / a
                elif node.kind == "sqrt":
                    local = 0.5 / np.sqrt(a)
                elif node.kind == "sin":
                    local = np.cos(a)
                elif node.kind == "cos":
                    local = -np.sin(a)
                elif node.kind == "tan":
                    local = 1.0 / np.cos(a) ** 2
                else:
                    local = 1.0 / (a * math.log(10.0))
                add_grad(i[0], grad * local)
            elif node.kind in {"add", "sub", "mul", "div", "pow"}:
                a = tape[i[0]].value
                b = tape[i[1]].value
                if node.kind == "add":
                    add_grad(i[0], grad)
                    add_grad(i[1], grad)
                elif node.kind == "sub":
                    add_grad(i[0], grad)
                    add_grad(i[1], -grad)
                elif node.kind == "mul":
                    add_grad(i[0], grad * b)
                    add_grad(i[1], grad * a)
                elif node.kind == "div":
                    add_grad(i[0], grad / b)
                    add_grad(i[1], -grad * a / b**2)
                else:
                    add_grad(i[0], grad * b * a ** (b - 1.0))
                    add_grad(i[1], grad * node.value * np.log(a))

        return gradients
