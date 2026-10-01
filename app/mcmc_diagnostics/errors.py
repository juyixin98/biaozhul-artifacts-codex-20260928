"""Exception translated to structured API failures."""

from __future__ import annotations


class DiagnosticError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
