"""Bridge package so `python -m p2.*` works from the repository root."""

from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "p2"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))
