"""AST-based mutation engine.

Generates mutants for three restricted operator families:
comparison operator swap, arithmetic operator swap and condition
negation. Mutants are produced by re-parsing the original source and
replacing exactly one node, so the original project is never touched.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

from .models import MutationOperator

# Alternative operators per original operator. The first alternative is
# the "nearest" semantic neighbour; all listed alternatives are emitted
# as separate mutants.
_COMPARE_ALT: dict[type, list[type]] = {
    ast.Eq: [ast.NotEq],
    ast.NotEq: [ast.Eq],
    ast.Lt: [ast.LtE, ast.Gt],
    ast.LtE: [ast.Lt, ast.GtE],
    ast.Gt: [ast.GtE, ast.Lt],
    ast.GtE: [ast.Gt, ast.LtE],
}

_ARITH_ALT: dict[type, list[type]] = {
    ast.Add: [ast.Sub],
    ast.Sub: [ast.Add],
    ast.Mult: [ast.Div],
    ast.Div: [ast.Mult],
    ast.FloorDiv: [ast.Div],
    ast.Mod: [ast.Mult],
}


@dataclass(frozen=True)
class MutationPoint:
    """One possible single-node mutation inside one file."""

    file_path: str          # path relative to the project root
    node_index: int         # index in the deterministic node walk
    operator: MutationOperator
    line: int
    col: int
    original_segment: str
    mutated_segment: str
    equivalent: bool        # statically detected equivalent mutant


def _terminates(stmt: ast.stmt) -> bool:
    return isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break))


def _unreachable_lines(tree: ast.AST) -> set[int]:
    """Lines belonging to statements after a terminator in the same block,
    plus bodies of `if False:`/`while False:` style literals."""
    dead: set[int] = set()

    class V(ast.NodeVisitor):
        def visit(self, node):  # noqa: ANN001
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if isinstance(block, list):
                    terminated = False
                    for stmt in block:
                        if terminated:
                            dead.update(
                                range(
                                    stmt.lineno,
                                    getattr(stmt, "end_lineno", stmt.lineno) + 1,
                                )
                            )
                        if _terminates(stmt):
                            terminated = True
            if isinstance(node, (ast.If, ast.While)):
                test = node.test
                if (
                    isinstance(test, ast.Constant)
                    and test.value is False
                ):
                    for stmt in node.body:
                        dead.update(
                            range(
                                stmt.lineno,
                                getattr(stmt, "end_lineno", stmt.lineno) + 1,
                            )
                        )
            self.generic_visit(node)

    V().visit(tree)
    return dead


def _walk_nodes(tree: ast.AST):
    """Deterministic pre-order walk used for stable node indexing."""
    for node in ast.walk(tree):
        yield node


def find_mutation_points(
    file_path: str,
    source: str,
    operators: set[MutationOperator],
) -> list[MutationPoint]:
    """Enumerate all mutation points of one source file."""
    tree = ast.parse(source)
    dead = _unreachable_lines(tree)
    points: list[MutationPoint] = []
    for idx, node in enumerate(_walk_nodes(tree)):
        equivalent = getattr(node, "lineno", -1) in dead
        if (
            MutationOperator.COMPARISON in operators
            and isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in _COMPARE_ALT
        ):
            original = ast.unparse(node)
            for alt in _COMPARE_ALT[type(node.ops[0])]:
                mutated = ast.unparse(
                    ast.Compare(
                        left=node.left, ops=[alt()], comparators=node.comparators
                    )
                )
                points.append(
                    MutationPoint(
                        file_path, idx, MutationOperator.COMPARISON,
                        node.lineno, node.col_offset, original, mutated,
                        equivalent,
                    )
                )
        if (
            MutationOperator.ARITHMETIC in operators
            and isinstance(node, ast.BinOp)
            and type(node.op) in _ARITH_ALT
        ):
            original = ast.unparse(node)
            for alt in _ARITH_ALT[type(node.op)]:
                mutated = ast.unparse(
                    ast.BinOp(left=node.left, op=alt(), right=node.right)
                )
                points.append(
                    MutationPoint(
                        file_path, idx, MutationOperator.ARITHMETIC,
                        node.lineno, node.col_offset, original, mutated,
                        equivalent,
                    )
                )
        if MutationOperator.CONDITION_NEGATION in operators and isinstance(
            node, (ast.If, ast.While)
        ):
            original = ast.unparse(node.test)
            mutated = ast.unparse(ast.UnaryOp(op=ast.Not(), operand=node.test))
            points.append(
                MutationPoint(
                    file_path, idx, MutationOperator.CONDITION_NEGATION,
                    node.lineno, node.col_offset, original, mutated,
                    equivalent,
                )
            )
    return points


def apply_mutation(source: str, point: MutationPoint) -> str:
    """Return the source with exactly the given mutation applied.

    Raises ValueError if the mutation point cannot be located, and lets
    SyntaxError from compile() propagate for invalid-syntax detection.
    """
    tree = ast.parse(source)
    nodes = list(_walk_nodes(tree))
    if point.node_index >= len(nodes):
        raise ValueError(f"node index {point.node_index} out of range")
    node = nodes[point.node_index]
    if point.operator is MutationOperator.COMPARISON and isinstance(
        node, ast.Compare
    ):
        target = ast.parse(point.mutated_segment, mode="eval").body
        node.ops = target.ops
    elif point.operator is MutationOperator.ARITHMETIC and isinstance(
        node, ast.BinOp
    ):
        target = ast.parse(point.mutated_segment, mode="eval").body
        node.op = target.op
    elif point.operator is MutationOperator.CONDITION_NEGATION and isinstance(
        node, (ast.If, ast.While)
    ):
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    else:
        raise ValueError("mutation point does not match node at index")
    ast.fix_missing_locations(tree)
    mutated = ast.unparse(tree)
    compile(mutated, point.file_path, "exec")  # raises SyntaxError if broken
    return mutated
