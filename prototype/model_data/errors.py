"""Structured model-data validation errors."""

from __future__ import annotations


class ModelDataError(ValueError):
    """A corpus or derived model example violates the shared data contract."""

    def __init__(self, code: str, detail: str, family_id: str | None = None):
        self.code = code
        self.detail = detail
        self.family_id = family_id
        prefix = f"{family_id}: " if family_id is not None else ""
        super().__init__(f"{code}: {prefix}{detail}")
