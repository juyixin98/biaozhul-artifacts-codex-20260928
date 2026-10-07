"""Deterministic AST source mutators."""

from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .models import MutationOperator


@dataclass(frozen=True)
class SourcePosition:
    path: str
    line: int
    column: int
    end_line: int | None
    end_column: int | None
    node_type: str


@dataclass(frozen=True)
class Mutation:
    mutant_id: str
    project_root: Path
    relative_path: str
    operator_family: MutationOperator
    before: str
    after: str
    location: SourcePosition
    original_source: str
    mutated_source: str


_COMPARISONS: dict[type[ast.cmpop], list[type[ast.cmpop]]] = {
    ast.Eq: [ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
    ast.NotEq: [ast.Eq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
    ast.Lt: [ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq],
    ast.LtE: [ast.Lt, ast.Gt, ast.GtE, ast.Eq, ast.NotEq],
    ast.Gt: [ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq],
    ast.GtE: [ast.Gt, ast.Lt, ast.LtE, ast.Eq, ast.NotEq],
}

_ARITHMETIC: dict[type[ast.operator], list[type[ast.operator]]] = {
    ast.Add: [ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod],
    ast.Sub: [ast.Add, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod],
    ast.Mult: [ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Mod],
    ast.Div: [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod],
    ast.FloorDiv: [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod],
    ast.Mod: [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv],
}

_OPERATOR_TEXT = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
}


def _op_name(op: ast.AST) -> str:
    return type(op).__name__


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    return segment if segment is not None else ast.unparse(node)


def _position(relative_path: str, node: ast.AST) -> SourcePosition:
    return SourcePosition(
        path=relative_path,
        line=getattr(node, "lineno", 0),
        column=getattr(node, "col_offset", 0),
        end_line=getattr(node, "end_lineno", None),
        end_column=getattr(node, "end_col_offset", None),
        node_type=type(node).__name__,
    )


def _mutant_id(
    relative_path: str,
    operator: MutationOperator,
    position: SourcePosition,
    before: str,
    after: str,
) -> str:
    digest = hashlib.sha256(
        "|".join(
            [
                relative_path,
                operator.value,
                str(position.line),
                str(position.column),
                position.node_type,
                before,
                after,
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"mut-{digest}"


def _make_mutation(
    *,
    project_root: Path,
    relative_path: str,
    source: str,
    tree: ast.Module,
    target: ast.AST,
    family: MutationOperator,
    before_node: ast.AST,
    after_node: ast.AST,
) -> Mutation:
    if isinstance(before_node, (ast.cmpop, ast.operator)):
        before = _OPERATOR_TEXT[type(before_node)]
        after = _OPERATOR_TEXT[type(after_node)]
    else:
        before = _source_segment(source, before_node)
        after = ast.unparse(after_node)
    position = _position(relative_path, target)
    mutated_tree = copy.deepcopy(tree)
    # The caller uses object identity on the original tree to find the target.
    target_id = id(target)
    original_by_id = {id(original): copied for original, copied in zip(ast.walk(tree), ast.walk(mutated_tree))}
    copied_target = original_by_id[target_id]
    _replace_target(copied_target, target, before_node, after_node)
    ast.fix_missing_locations(mutated_tree)
    mutated_source = ast.unparse(mutated_tree)
    return Mutation(
        mutant_id=_mutant_id(relative_path, family, position, before, after),
        project_root=project_root,
        relative_path=relative_path,
        operator_family=family,
        before=before,
        after=after,
        location=position,
        original_source=source,
        mutated_source=mutated_source,
    )


def _replace_target(
    copied_target: ast.AST,
    original_target: ast.AST,
    before_node: ast.AST,
    after_node: ast.AST,
) -> None:
    if isinstance(copied_target, ast.Compare) and isinstance(original_target, ast.Compare):
        index = original_target.ops.index(before_node)
        copied_target.ops[index] = copy.copy(after_node)
        return
    if isinstance(copied_target, (ast.BinOp, ast.AugAssign)) and isinstance(
        original_target, (ast.BinOp, ast.AugAssign)
    ):
        copied_target.op = copy.copy(after_node)
        return
    if isinstance(copied_target, ast.If) and isinstance(original_target, ast.If):
        copied_target.test = after_node
        return
    if isinstance(copied_target, ast.UnaryOp) and isinstance(original_target, ast.UnaryOp):
        copied_target.operand = after_node
        return
    raise AssertionError(f"unsupported mutation target {type(copied_target).__name__}")


def _inverted_test(test: ast.expr) -> ast.expr:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return copy.deepcopy(test.operand)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op_type = type(test.ops[0])
        inverse = {
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.Lt: ast.GtE,
            ast.LtE: ast.Gt,
            ast.Gt: ast.LtE,
            ast.GtE: ast.Lt,
        }.get(op_type)
        if inverse is not None:
            result = copy.deepcopy(test)
            result.ops = [copy.copy(inverse())]
            return result
    return ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(test))


def mutate_file(
    project_root: Path,
    absolute_path: Path,
    operators: set[MutationOperator],
) -> list[Mutation]:
    relative_path = absolute_path.relative_to(project_root).as_posix()
    source = absolute_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(absolute_path))
    mutations: list[Mutation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and MutationOperator.COMPARISON in operators:
            for op in node.ops:
                for replacement_type in _COMPARISONS.get(type(op), []):
                    mutations.append(
                        _make_mutation(
                            project_root=project_root,
                            relative_path=relative_path,
                            source=source,
                            tree=tree,
                            target=node,
                            family=MutationOperator.COMPARISON,
                            before_node=op,
                            after_node=replacement_type(),
                        )
                    )

        if isinstance(node, (ast.BinOp, ast.AugAssign)) and MutationOperator.ARITHMETIC in operators:
            for replacement_type in _ARITHMETIC.get(type(node.op), []):
                mutations.append(
                    _make_mutation(
                        project_root=project_root,
                        relative_path=relative_path,
                        source=source,
                        tree=tree,
                        target=node,
                        family=MutationOperator.ARITHMETIC,
                        before_node=node.op,
                        after_node=replacement_type(),
                    )
                )

        if isinstance(node, ast.If) and MutationOperator.CONDITIONAL_INVERT in operators:
            original_test = node.test
            mutations.append(
                _make_mutation(
                    project_root=project_root,
                    relative_path=relative_path,
                    source=source,
                    tree=tree,
                    target=node,
                    family=MutationOperator.CONDITIONAL_INVERT,
                    before_node=original_test,
                    after_node=_inverted_test(original_test),
                )
            )

        if isinstance(node, ast.If) and MutationOperator.CONDITION_ERASE in operators:
            original_test = node.test
            mutations.append(
                _make_mutation(
                    project_root=project_root,
                    relative_path=relative_path,
                    source=source,
                    tree=tree,
                    target=node,
                    family=MutationOperator.CONDITION_ERASE,
                    before_node=original_test,
                    after_node=ast.Name(id="if", ctx=ast.Load()),
                )
            )

    return mutations


def select_source_files(
    project_root: Path,
    source_patterns: list[str],
    exclude_patterns: list[str],
) -> list[Path]:
    selected: set[Path] = set()
    excluded: set[Path] = set()
    for pattern in exclude_patterns:
        excluded.update(project_root.glob(pattern))
    for pattern in source_patterns:
        selected.update(project_root.glob(pattern))
    files = [
        path
        for path in selected
        if path.is_file()
        and path.suffix == ".py"
        and path not in excluded
        and ".git" not in path.relative_to(project_root).parts
    ]
    return sorted(files, key=lambda path: path.relative_to(project_root).as_posix())


def generate_mutations(
    project_root: Path,
    source_patterns: list[str],
    exclude_patterns: list[str],
    operators: list[MutationOperator],
) -> list[Mutation]:
    operator_set = set(operators)
    result: list[Mutation] = []
    for path in select_source_files(project_root, source_patterns, exclude_patterns):
        result.extend(mutate_file(project_root, path, operator_set))
    result.sort(key=lambda mutation: (mutation.relative_path, mutation.location.line, mutation.mutant_id))
    return result
