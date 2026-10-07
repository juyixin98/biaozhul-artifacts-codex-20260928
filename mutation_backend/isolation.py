"""Immutable project snapshots and per-mutant workspace copies."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from .models import SourceFileEvidence


IGNORED_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


@dataclass(frozen=True)
class Snapshot:
    root: Path
    files: tuple[SourceFileEvidence, ...]


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name in IGNORED_DIRECTORIES}
    ignored.update(name for name in names if name.endswith(".pyc"))
    return ignored


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(project_root: Path) -> tuple[SourceFileEvidence, ...]:
    files = []
    for path in sorted(project_root.rglob("*.py")):
        if any(part in IGNORED_DIRECTORIES for part in path.relative_to(project_root).parts):
            continue
        files.append(
            SourceFileEvidence(
                path=path.relative_to(project_root).as_posix(),
                sha256=hash_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(files)


def copy_project(project_root: Path, destination: Path) -> Snapshot:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project_root, destination, ignore=_ignore)
    return Snapshot(root=destination, files=source_snapshot(destination))


def remove_workspace(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
