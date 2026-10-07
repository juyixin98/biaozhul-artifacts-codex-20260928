"""Conservative static equivalence checks.

An equivalent mutant is intentionally reported as undetermined. Static evidence
explains why the mutant cannot change behavior, but the mutation result is not
counted as a survived mutant or as evidence that tests are merely insufficient.
"""

from __future__ import annotations

import ast
import copy

from .models import MutationOperator
from .mutator import Mutation, _OPERATOR_TEXT, _source_segment


_OP_BY_SYMBOL = {text: op_type for op_type, text in _OPERATOR_TEXT.items()}


class _Uncertain(RuntimeError):
    """Raised when static evaluation cannot prove equivalence."""


class _StaticEvaluator:
    def evaluate(self, node: ast.AST):
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            return self._compare(node)
        raise _Uncertain("expression contains a non-literal value")

    def _binop(self, node: ast.BinOp):
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
        except (ArithmeticError, TypeError) as exc:
            raise _Uncertain(str(exc)) from exc
        raise _Uncertain("unsupported arithmetic operator")

    def _compare(self, node: ast.Compare):
        left = self.evaluate(node.left)
        right = self.evaluate(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        raise _Uncertain("unsupported comparison operator")


def _safe_expression(node: ast.expr) -> ast.Expression:
    for child in ast.walk(node):
        if isinstance(child, (ast.Call, ast.Attribute, ast.Name, ast.Subscript)):
            raise _Uncertain("expression may depend on runtime state")
    copied = ast.Expression(body=ast.copy_location(copy.deepcopy(node), node))
    ast.fix_missing_locations(copied)
    return copied


def _constant_value(node: ast.expr):
    expression = _safe_expression(node)
    return _StaticEvaluator().evaluate(expression)


def _truth(value) -> bool:
    return bool(value)


def _node_at(tree: ast.Module, mutation: Mutation) -> ast.AST | None:
    loc = mutation.location
    for node in ast.walk(tree):
        if (
            getattr(node, "lineno", None) == loc.line
            and getattr(node, "col_offset", None) == loc.column
            and type(node).__name__ == loc.node_type
        ):
            if isinstance(node, (ast.BinOp, ast.AugAssign, ast.Compare)):
                if mutation.operator_family == MutationOperator.ARITHMETIC and _OPERATOR_TEXT.get(
                    type(node.op)
                ) != mutation.before:
                    continue
                if mutation.operator_family == MutationOperator.COMPARISON and not any(
                    _OPERATOR_TEXT.get(type(op)) == mutation.before for op in node.ops
                ):
                    continue
            if isinstance(node, ast.If) and _source_segment(mutation.original_source, node.test) != mutation.before:
                continue
            return node
    return None


def equivalence_basis(mutation: Mutation) -> str | None:
    """Return a human-readable proof basis, or None when not proven equal."""

    try:
        tree = ast.parse(mutation.original_source, filename=mutation.relative_path)
        node = _node_at(tree, mutation)
        if node is None:
            return None

        if mutation.operator_family == MutationOperator.ARITHMETIC and isinstance(node, ast.BinOp):
            replacement_type = _OP_BY_SYMBOL[mutation.after]
            changed = ast.BinOp(left=node.left, op=replacement_type(), right=node.right)
            original_value = _constant_value(node)
            mutated_value = _constant_value(changed)
            if original_value == mutated_value and type(original_value) is type(mutated_value):
                return (
                    f"constant-folded expression has identical value: "
                    f"{mutation.before} and {mutation.after} both evaluate to {original_value!r}"
                )

        if mutation.operator_family == MutationOperator.ARITHMETIC and isinstance(node, ast.AugAssign):
            return None

        if mutation.operator_family == MutationOperator.COMPARISON and isinstance(node, ast.Compare):
            replacement_type = _OP_BY_SYMBOL[mutation.after]
            changed = ast.Compare(left=node.left, ops=[replacement_type()], comparators=node.comparators)
            original_value = _constant_value(node)
            mutated_value = _constant_value(changed)
            if original_value == mutated_value:
                return (
                    f"constant comparison has identical truth value: "
                    f"{mutation.before} and {mutation.after} both yield {original_value!r}"
                )

        if mutation.operator_family == MutationOperator.CONDITIONAL_INVERT and isinstance(node, ast.If):
            mutated_test = ast.parse(mutation.after, mode="eval")
            if _truth(_constant_value(node.test)) == _truth(_constant_value(mutated_test.body)):
                return "constant condition and its inversion select the same branch"

    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError, _Uncertain):
        return None
    return None
