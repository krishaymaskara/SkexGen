"""Stable structured errors for the GE1 package boundary."""

from __future__ import annotations


class GraphEncoderError(ValueError):
    """A GE1 configuration or access request violates the frozen contract."""

    def __init__(self, code, detail):
        if not isinstance(code, str) or not code:
            raise ValueError("GraphEncoderError code must be a nonempty string")
        if not isinstance(detail, str) or not detail:
            raise ValueError("GraphEncoderError detail must be a nonempty string")
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))
