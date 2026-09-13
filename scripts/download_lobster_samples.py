"""Download the public 2012-06-21 level-10 LOBSTER samples."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from p2.lobster_samples import LOBSTER_SAMPLE_SYMBOLS, download_samples


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", choices=LOBSTER_SAMPLE_SYMBOLS, default=list(LOBSTER_SAMPLE_SYMBOLS))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("P2_DATA_DIR", REPO_ROOT / "data")) / "lobster",
    )
    args = parser.parse_args()
    download_samples(args.symbols, args.data_dir)


if __name__ == "__main__":
    main()
