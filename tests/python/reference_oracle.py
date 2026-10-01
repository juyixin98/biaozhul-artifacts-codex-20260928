#!/usr/bin/env python3
"""Exhaustive reference oracle for small directed path covers.

The Rust implementation never supplies an expected answer to this script. It
enumerates every ordered permutation of vertices and every way to split each
permutation into contiguous paths. A split is accepted exactly when every
adjacent pair in every resulting path is a graph edge.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any


def load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def enumerate_path_covers(document: dict[str, Any]) -> dict[str, Any]:
    graph = document["graph"]
    labels = list(graph["vertices"])
    edges = {(edge["from"], edge["to"]) for edge in graph.get("edges", [])}
    vertex_count = len(labels)

    covers: set[tuple[tuple[str, ...], ...]] = set()
    split_count = 1 << max(vertex_count - 1, 0)
    checked_splits = 0
    total = factorial(vertex_count)

    for perm_index, permutation in enumerate(itertools.permutations(range(vertex_count)), start=1):
        ordered_labels = [labels[index] for index in permutation]
        for mask in range(split_count):
            checked_splits += 1
            paths: list[list[str]] = [[]]
            valid = True
            for index, vertex in enumerate(ordered_labels):
                paths[-1].append(vertex)
                if index < vertex_count - 1:
                    next_vertex = ordered_labels[index + 1]
                    keep_together = ((mask >> index) & 1) == 1
                    if keep_together:
                        if (vertex, next_vertex) not in edges:
                            valid = False
                            break
                    else:
                        paths.append([])
            if valid:
                canonical = tuple(sorted(tuple(path) for path in paths))
                covers.add(canonical)
        if perm_index % 1000 == 0 or perm_index == total:
            print(
                f"progress permutations={perm_index}/{total} accepted_covers={len(covers)}",
                file=sys.stderr,
            )

    minimum = min((len(cover) for cover in covers), default=vertex_count)
    minimum_covers = sorted(
        ([list(path) for path in cover] for cover in covers if len(cover) == minimum),
        key=lambda cover: json.dumps(cover, sort_keys=True),
    )

    fingerprint_input = json.dumps(
        {"vertices": labels, "edges": sorted(edges)}, sort_keys=True
    ).encode("utf-8")
    run_id = "pyref-" + hashlib.sha256(fingerprint_input).hexdigest()[:16]

    return {
        "run_id": run_id,
        "name": document["name"],
        "vertex_count": vertex_count,
        "edge_count": len(edges),
        "checked_permutations": total,
        "checked_splits": checked_splits,
        "accepted_cover_count": len(covers),
        "minimum_path_count": minimum,
        "minimum_covers": minimum_covers,
    }


def factorial(value: int) -> int:
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: reference_oracle.py FIXTURE.json [...]", file=sys.stderr)
        return 2

    results = {}
    for argument in argv:
        path = Path(argument)
        document = load_fixture(path)
        run_id = "pyref-" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
        print(
            f"{run_id} input={path} name={document['name']} step=load status=ok",
            file=sys.stderr,
        )
        result = enumerate_path_covers(document)
        print(
            f"{result['run_id']} input={path} name={document['name']} "
            f"step=enumerate status=ok minimum={result['minimum_path_count']} "
            f"splits={result['checked_splits']}",
            file=sys.stderr,
        )
        results[document["name"]] = result

    json.dump(results, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
