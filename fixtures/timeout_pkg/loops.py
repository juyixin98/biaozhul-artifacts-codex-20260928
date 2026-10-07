"""Fixture where specific mutants spin forever (timeout exercise)."""


def countdown(n):
    while n > 0:
        n = n - 1
    return n
