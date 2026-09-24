#!/usr/bin/env python3
"""Regenerate the published figures from the published tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from p2.research_outputs import FIGURE_FILES, render_research_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--published-dir",
        default=Path(__file__).resolve().parents[1] / "results" / "published",
        type=Path,
    )
    arguments = parser.parse_args()
    render_research_figures(arguments.published_dir)
    for name in FIGURE_FILES:
        print(arguments.published_dir / name)


if __name__ == "__main__":
    main()
