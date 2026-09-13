"""Download helpers for the public LOBSTER sample archives."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


LOBSTER_SAMPLE_DATE = "2012-06-21"
LOBSTER_SAMPLE_LEVEL = 10
LOBSTER_SAMPLE_SYMBOLS = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")
LOBSTER_SAMPLE_BASE_URL = "https://data.lobsterdata.com/info/sample"


def sample_archive_url(
    symbol: str,
    date: str = LOBSTER_SAMPLE_DATE,
    level: int = LOBSTER_SAMPLE_LEVEL,
) -> str:
    normalized = symbol.upper()
    if normalized not in LOBSTER_SAMPLE_SYMBOLS:
        supported = ", ".join(LOBSTER_SAMPLE_SYMBOLS)
        raise ValueError(f"Unsupported sample symbol {normalized!r}; choose one of {supported}.")
    return f"{LOBSTER_SAMPLE_BASE_URL}/LOBSTER_SampleFile_{normalized}_{date}_{level}.zip"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_zip(archive_path: Path, output_dir: Path) -> list[Path]:
    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        for member in members:
            destination = (output_dir / member.filename).resolve()
            if not destination.is_relative_to(output_root):
                raise ValueError(f"Archive member escapes the data directory: {member.filename}")
        archive.extractall(output_dir)
    return [(output_dir / member.filename).resolve() for member in members]


def download_sample(symbol: str, output_dir: str | Path) -> list[tuple[Path, str]]:
    """Download and extract one public sample, returning file hashes."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    url = sample_archive_url(symbol)
    with tempfile.TemporaryDirectory() as temporary_dir:
        archive_path = Path(temporary_dir) / Path(url).name
        with urllib.request.urlopen(url, timeout=60) as response, archive_path.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        archive_hash = sha256_file(archive_path)
        print(f"sha256 {archive_hash} {archive_path.name}")
        extracted = _extract_zip(archive_path, output)

    hashed_files = [(path, sha256_file(path)) for path in sorted(extracted)]
    for path, digest in hashed_files:
        print(f"sha256 {digest} {path.name}")
    return hashed_files


def download_samples(symbols: Iterable[str], output_dir: str | Path) -> list[tuple[Path, str]]:
    hashes: list[tuple[Path, str]] = []
    for symbol in symbols:
        hashes.extend(download_sample(symbol, output_dir))
    return hashes
