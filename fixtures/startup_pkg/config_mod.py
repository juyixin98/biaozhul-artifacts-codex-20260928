"""Fixture where one arithmetic mutant breaks module import."""

STEP = 10 // 2
VALUES = list(range(STEP))


def values():
    return VALUES
