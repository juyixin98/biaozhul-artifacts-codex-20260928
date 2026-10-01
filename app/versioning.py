"""Central version metadata for response and persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings

SCHEMA_MIGRATION_VERSION = 1


@dataclass(frozen=True)
class VersionInfo:
    schema_version: str
    service_version: str
    kernel_version: str
    generator_version: str
    schema_migration_version: int


def current_versions(settings: Settings) -> VersionInfo:
    return VersionInfo(
        schema_version=settings.schema_version,
        service_version=settings.service_version,
        kernel_version=settings.kernel_version,
        generator_version=settings.generator_version,
        schema_migration_version=SCHEMA_MIGRATION_VERSION,
    )
