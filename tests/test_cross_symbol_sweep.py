from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

from p2.baselines import SymmetricMM
from p2.cross_symbol_sweep import DEFAULT_DATE, run_symbol_sweep


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _write_aapl_fixture(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    copyfile(
        FIXTURE_DIR / "lobster_message.csv",
        data_dir / "AAPL_2012-06-21_34200000_57600000_message_10.csv",
    )
    copyfile(
        FIXTURE_DIR / "lobster_orderbook.csv",
        data_dir / "AAPL_2012-06-21_34200000_57600000_orderbook_10.csv",
    )


def test_cross_symbol_sweep_missing_data_skips(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "lobster"
    results_dir = tmp_path / "results"
    _write_aapl_fixture(data_dir)

    results = run_symbol_sweep(
        symbols=["AAPL", "AMZN", "GOOG", "INTC", "MSFT", "SPY"],
        date=DEFAULT_DATE,
        data_dir=data_dir,
        results_dir=results_dir,
        quoters={"symmetric": SymmetricMM(half_spread=0.05)},
    )

    assert results["symbol"].tolist() == ["AAPL"]
    assert results.attrs["symbols_run"] == ["AAPL"]
    assert results.attrs["symbols_skipped"] == ["AMZN", "GOOG", "INTC", "MSFT", "SPY"]


def test_cross_symbol_sweep_writes_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "lobster"
    results_dir = tmp_path / "results"
    _write_aapl_fixture(data_dir)

    results = run_symbol_sweep(
        symbols=["AAPL", "AMZN", "GOOG", "INTC", "MSFT", "SPY"],
        date=DEFAULT_DATE,
        data_dir=data_dir,
        results_dir=results_dir,
        quoters={"symmetric": SymmetricMM(half_spread=0.05)},
    )

    assert len(results) == 1
    assert (results_dir / "cross_symbol_ablation.csv").exists()
    assert (results_dir / "cross_symbol_ablation.parquet").exists()
    assert (results_dir / "summary.json").exists()

    summary = json.loads((results_dir / "summary.json").read_text())
    assert summary["symbols_run"] == ["AAPL"]
    assert summary["symbols_skipped"] == ["AMZN", "GOOG", "INTC", "MSFT", "SPY"]
    assert summary["quoters"] == ["symmetric"]
    assert summary["total_simulations"] == 1
