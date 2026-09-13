from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from p2.lobster_samples import _extract_zip, sample_archive_url, sha256_file


def test_sample_archive_url_uses_official_host() -> None:
    assert sample_archive_url("aapl") == (
        "https://data.lobsterdata.com/info/sample/"
        "LOBSTER_SampleFile_AAPL_2012-06-21_10.zip"
    )


def test_extract_zip_returns_hashable_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample.zip"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    payload = b"sample-data\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("AAPL_message_10.csv", payload)

    extracted = _extract_zip(archive_path, output_dir)

    assert extracted == [(output_dir / "AAPL_message_10.csv").resolve()]
    assert sha256_file(extracted[0]) == hashlib.sha256(payload).hexdigest()


def test_extract_zip_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.csv", "unsafe")

    with pytest.raises(ValueError, match="escapes"):
        _extract_zip(archive_path, output_dir)
