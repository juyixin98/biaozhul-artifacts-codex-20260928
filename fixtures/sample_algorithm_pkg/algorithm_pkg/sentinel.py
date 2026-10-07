"""Import-time startup behavior."""


def _invariant() -> bool:
    return len("a") == 1


if not _invariant():
    raise RuntimeError("package sentinel failed during startup")


def boot_sentinel() -> str:
    return "ready"
