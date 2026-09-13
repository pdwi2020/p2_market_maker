"""Cross-symbol LOBSTER sweep for quoter ablations."""

from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from p2.baselines import _simulate_quotes
from p2.calibration import calibrate_from_replay_files
from p2.config import AdverseSelectionConfig
from p2.lobster_samples import LOBSTER_SAMPLE_SYMBOLS, _extract_zip, sample_archive_url
from p2.lobster_replay import BacktestResult, LOBSTERReplayer


DEFAULT_DATE = "2012-06-21"
DEFAULT_LEVEL = 10
DEFAULT_T = 1.0
DEFAULT_DT = 0.001
DEFAULT_GAMMA = 0.1
DEFAULT_REPLAY_T = 23_400.0
DEFAULT_REPLAY_GAMMA = 0.0001
DEFAULT_Q_MAX = 10
DEFAULT_N_PATHS = 1000

# LOBSTER sample downloads are single archives containing both files. The tuple
# repeats the archive slot for compatibility with the existing resolver.
LOBSTER_FREE_SAMPLE_URLS: dict[str, tuple[str | None, str | None]] = {
    symbol: (sample_archive_url(symbol), sample_archive_url(symbol))
    for symbol in LOBSTER_SAMPLE_SYMBOLS
}


def _sample_paths(symbol: str, date: str, level: int, data_dir: Path) -> tuple[Path, Path]:
    prefix = f"{symbol}_{date}_34200000_57600000"
    return (
        data_dir / f"{prefix}_message_{level}.csv",
        data_dir / f"{prefix}_orderbook_{level}.csv",
    )


def _manual_download_message(symbol: str, date: str, level: int, data_dir: Path) -> str:
    message_path, orderbook_path = _sample_paths(symbol, date, level, data_dir)
    return (
        f"LOBSTER sample for {symbol} {date} L{level} is missing. "
        f"Expected files: '{message_path.name}' and '{orderbook_path.name}' in {data_dir}. "
        "Automatic download was unavailable or failed. See README.md and place the extracted CSVs under data/lobster/."
    )


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    if archive_path.suffix == ".zip":
        _extract_zip(archive_path, output_dir)
        return
    if archive_path.suffix == ".7z":
        tool = shutil.which("7zz") or shutil.which("7z")
        if tool is None:
            raise RuntimeError("7z extraction tool not available.")
        subprocess.run([tool, "x", str(archive_path), f"-o{output_dir}", "-y"], check=True, capture_output=True, text=True)
        return
    raise RuntimeError(f"Unsupported archive format: {archive_path.suffix}")


def ensure_lobster_data(
    symbol: str,
    date: str,
    level: int,
    data_dir: Path,
    allow_download: bool = False,
) -> tuple[Path, Path]:
    """Resolve local LOBSTER CSVs, optionally attempting archive download."""

    message_path, orderbook_path = _sample_paths(symbol, date, level, data_dir)
    if message_path.exists() and orderbook_path.exists():
        return message_path, orderbook_path
    if not allow_download:
        raise FileNotFoundError(_manual_download_message(symbol, date, level, data_dir))

    archive_urls = {url for url in LOBSTER_FREE_SAMPLE_URLS.get(symbol, (None, None)) if url}
    if not archive_urls:
        raise FileNotFoundError(_manual_download_message(symbol, date, level, data_dir))

    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for url in archive_urls:
                archive_path = tmp_path / Path(urlparse(url).path).name
                urllib.request.urlretrieve(url, archive_path)
                _extract_archive(archive_path, data_dir)
    except Exception as error:  # pragma: no cover - network path is best effort.
        raise FileNotFoundError(_manual_download_message(symbol, date, level, data_dir)) from error

    if message_path.exists() and orderbook_path.exists():
        return message_path, orderbook_path
    raise FileNotFoundError(_manual_download_message(symbol, date, level, data_dir))


def _instantiate_quoter(spec: Any, calibration: dict[str, float | int | str]) -> Any:
    if hasattr(spec, "quotes"):
        if is_dataclass(spec):
            kwargs = {field.name: getattr(spec, field.name) for field in fields(spec) if field.init}
            return type(spec)(**kwargs)
        return spec
    if callable(spec):
        params = inspect.signature(spec).parameters
        instance = spec(calibration) if params else spec()
        if hasattr(instance, "quotes"):
            return instance
    raise TypeError(f"Unsupported quoter specification: {type(spec)!r}")


def _replay_metrics(result: BacktestResult) -> dict[str, float]:
    inventory = np.asarray(result.inventory_path, dtype=float)
    return {
        "replay_terminal_pnl": float(result.pnl),
        "replay_quoted_width": float(result.quoted_width),
        "replay_realized_spread": float(result.realized_spread),
        "replay_avg_abs_inventory": float(np.mean(np.abs(inventory))) if inventory.size else 0.0,
        "replay_fill_count": float(result.fill_quantity),
    }


def _synthetic_metrics(
    quoter: Any,
    calibration: dict[str, float | int | str],
    *,
    initial_mid: float,
    seed: int,
) -> dict[str, float]:
    gamma = float(getattr(quoter, "gamma", DEFAULT_GAMMA))
    horizon = DEFAULT_T
    q_max = int(getattr(quoter, "Q_max", DEFAULT_Q_MAX))
    synthetic_quoter = _with_horizon(quoter, horizon)
    result = _simulate_quotes(
        synthetic_quoter,
        n_paths=DEFAULT_N_PATHS,
        sigma=float(calibration["sigma"]),
        gamma=gamma,
        kappa=float(calibration["kappa"]),
        A=float(calibration["A"]),
        T=horizon,
        dt=DEFAULT_DT,
        Q_max=q_max,
        seed=seed,
        s0=initial_mid,
        adverse_selection=AdverseSelectionConfig(enabled=True, epsilon=float(calibration["epsilon"])),
    )
    return {
        "terminal_pnl": float(result.mean_pnl),
        "sharpe": float(result.sharpe),
        "spread_capture": float(result.spread_capture),
        "avg_abs_inventory": float(result.avg_abs_inventory),
        "fill_rate_bid": float(result.avg_bid_fill_rate),
        "fill_rate_ask": float(result.avg_ask_fill_rate),
    }


def _with_horizon(quoter: Any, horizon: float) -> Any:
    if not is_dataclass(quoter):
        return quoter
    kwargs = {field.name: getattr(quoter, field.name) for field in fields(quoter) if field.init}
    if "T" not in kwargs:
        return quoter
    kwargs["T"] = horizon
    return type(quoter)(**kwargs)


def run_symbol_sweep(
    symbols: list[str],
    date: str,
    data_dir: Path,
    results_dir: Path,
    quoters: dict[str, Any],
) -> pd.DataFrame:
    """Run a calibrated symbol sweep across the provided quoter set."""

    rows: list[dict[str, float | str]] = []
    symbols_run: list[str] = []
    symbols_skipped: list[str] = []
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for symbol_idx, raw_symbol in enumerate(symbols):
        symbol = raw_symbol.upper()
        try:
            message_path, orderbook_path = ensure_lobster_data(symbol, date, DEFAULT_LEVEL, data_dir, allow_download=False)
        except FileNotFoundError:
            symbols_skipped.append(symbol)
            continue

        calibration = calibrate_from_replay_files(orderbook_path, message_path)
        replayer = LOBSTERReplayer().load(orderbook_file=orderbook_path, message_file=message_path)
        initial_mid = float(replayer.mid_price_series().iloc[0])
        symbols_run.append(symbol)

        for quoter_idx, (quoter_name, quoter_spec) in enumerate(quoters.items()):
            quoter = _instantiate_quoter(quoter_spec, calibration)
            replay_result = replayer.run_strategy(
                quoter.quotes,
                session_duration_seconds=DEFAULT_REPLAY_T,
                inventory_limit=DEFAULT_Q_MAX,
            )
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "quoter": quoter_name,
                    "sigma": float(calibration["sigma"]),
                    "A": float(calibration["A"]),
                    "kappa": float(calibration["kappa"]),
                    "epsilon": float(calibration["epsilon"]),
                    **_synthetic_metrics(
                        quoter,
                        calibration,
                        initial_mid=initial_mid,
                        seed=42 + symbol_idx * 1_000 + quoter_idx,
                    ),
                    **_replay_metrics(replay_result),
                }
            )

    results = pd.DataFrame(rows).sort_values(["symbol", "quoter"]).reset_index(drop=True) if rows else pd.DataFrame(
        columns=[
            "symbol",
            "date",
            "quoter",
            "sigma",
            "A",
            "kappa",
            "epsilon",
            "terminal_pnl",
            "sharpe",
            "spread_capture",
            "avg_abs_inventory",
            "fill_rate_bid",
            "fill_rate_ask",
            "replay_terminal_pnl",
            "replay_quoted_width",
            "replay_realized_spread",
            "replay_avg_abs_inventory",
            "replay_fill_count",
        ]
    )
    results.to_csv(results_dir / "cross_symbol_ablation.csv", index=False)
    results.to_parquet(results_dir / "cross_symbol_ablation.parquet", index=False)

    summary = {
        "symbols_run": symbols_run,
        "symbols_skipped": symbols_skipped,
        "quoters": list(quoters.keys()),
        "total_simulations": len(rows),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    results.attrs["symbols_run"] = symbols_run
    results.attrs["symbols_skipped"] = symbols_skipped
    return results
