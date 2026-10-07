"""Input model: structured documents, syntax constraints, verdicts."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum

from pydantic import BaseModel, Field

SETTING_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[^=]*$")

# Keys whose values are treated as sensitive and masked in diagnostics.
SENSITIVE_KEYS = {"password", "secret", "token", "api_key"}


class Verdict(str, Enum):
    """Failure classification of one execution of the target program."""

    TARGET = "target"  # the specific error we want to preserve
    OTHER = "other"  # a different error
    PASS = "pass"  # ran successfully
    INDETERMINATE = "indeterminate"  # timeout / cannot decide


class Block(BaseModel):
    """One block of a structured document.

    kind: 'header' | 'setting' | 'footer'
    text: raw content of the block.
    """

    kind: str
    text: str

    def masked_text(self) -> str:
        if self.kind == "setting" and "=" in self.text:
            key = self.text.split("=", 1)[0].strip().lower()
            if key in SENSITIVE_KEYS:
                return f"{key}=***"
        return self.text


class Document(BaseModel):
    """A structured input: ordered blocks with declared syntax constraints.

    Constraints (declared, enforced by `validate_syntax`):
      1. exactly one header block, and it is first;
      2. exactly one footer block, and it is last;
      3. every setting block text matches `key=value`.
    """

    blocks: list[Block] = Field(min_length=2)

    def validate_syntax(self) -> list[str]:
        errors: list[str] = []
        kinds = [b.kind for b in self.blocks]
        if kinds[0] != "header":
            errors.append("first block must be a header")
        if kinds[-1] != "footer":
            errors.append("last block must be a footer")
        if kinds.count("header") != 1:
            errors.append("document must contain exactly one header")
        if kinds.count("footer") != 1:
            errors.append("document must contain exactly one footer")
        for i, b in enumerate(self.blocks):
            if b.kind == "setting" and not SETTING_RE.match(b.text):
                errors.append(f"block {i}: setting must match key=value")
            if b.kind not in ("header", "setting", "footer"):
                errors.append(f"block {i}: unknown kind {b.kind!r}")
        return errors

    def removable_indexes(self) -> list[int]:
        """Indexes of blocks the reduction operator may delete.

        Header and footer are structural and never removable; only
        setting blocks may be removed, which keeps every candidate
        syntactically valid by construction.
        """
        return [i for i, b in enumerate(self.blocks) if b.kind == "setting"]

    def without(self, index: int) -> "Document":
        return Document(blocks=[b for i, b in enumerate(self.blocks) if i != index])

    def to_text(self) -> str:
        return "\n".join(b.text for b in self.blocks) + "\n"

    def canonical(self) -> str:
        """Stable serialization used for cache keys and digests."""
        return json.dumps(
            [[b.kind, b.text] for b in self.blocks], separators=(",", ":")
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def masked_summary(self) -> str:
        return " | ".join(b.masked_text() for b in self.blocks)

    @classmethod
    def from_text(cls, text: str) -> "Document":
        """Parse the canonical text form: first line header, last line
        footer, middle lines settings."""
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            raise ValueError("document needs at least header and footer lines")
        blocks = [Block(kind="header", text=lines[0])]
        blocks += [Block(kind="setting", text=ln) for ln in lines[1:-1]]
        blocks.append(Block(kind="footer", text=lines[-1]))
        return cls(blocks=blocks)
