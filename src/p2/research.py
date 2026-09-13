"""End-to-end execution for the preregistered research study."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from typing import Sequence

from p2.config import P2Config, load_config
from p2.data.bybit import find_bybit_files, reconstruct_bybit_day
from p2.data.mbo import find_mbo_file
from p2.hft_crosscheck import CROSSCHECK_DATES, run_hft_crosscheck
from p2.lobster_appendix import run_lobster_appendix
from p2.queue_validation import VALIDATION_DATES, validate_es_queues
from p2.research_outputs import publish_research_outputs
from p2.research_study import (
    SELECTION_START,
    TEST_END,
    TEST_START,
    eligible_dates,
    run_crypto_study,
)


CRYPTO_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)


@dataclass(frozen=True)
class ResearchResult:
    """Published location and numeric execution summary."""

    output_dir: Path
    summary: dict[str, object]


def _date_range(start: date, end: date) -> tuple[date, ...]:
    return tuple(
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    )


def available_bybit_dates(source_root: str | Path, symbol: str) -> tuple[date, ...]:
    """Discover dates having both raw book and trade files."""
    root = Path(source_root)
    layouts = (
        (
            root / "samples" / "bybit" / symbol / "orderbook_l2",
            root / "samples" / "bybit" / symbol / "trades",
        ),
        (
            root / "exchange=bybit" / "instrument_type=orderbook_l2" / symbol,
            root / "exchange=bybit" / "instrument_type=trades" / symbol,
        ),
    )
    discovered: set[date] = set()
    for book_root, trade_root in layouts:
        book_dates = {
            path.parent.name.removeprefix("date=")
            for path in book_root.glob("date=*/orderbook.parquet")
        }
        trade_dates = {
            path.parent.name.removeprefix("date=")
            for path in trade_root.glob("date=*/trades.parquet")
        }
        for value in book_dates & trade_dates:
            try:
                discovered.add(date.fromisoformat(value))
            except ValueError:
                continue
    return tuple(sorted(discovered))


def required_bybit_dates(
    available: dict[str, Sequence[date]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return the minimal raw-data schedule needed by all crypto studies."""
    if available is None:
        available = {
            "BTCUSDT": _date_range(SELECTION_START, TEST_END),
            "ETHUSDT": _date_range(TEST_START - timedelta(days=1), TEST_END),
            "SOLUSDT": _date_range(TEST_START - timedelta(days=1), TEST_END),
        }
    btc_targets = set(
        eligible_dates(
            available.get("BTCUSDT", ()),
            SELECTION_START,
            TEST_END,
        )
    )
    crosscheck_days = {date.fromisoformat(value) for value in CROSSCHECK_DATES}
    missing_crosscheck = crosscheck_days - btc_targets
    if missing_crosscheck:
        missing = [value.isoformat() for value in sorted(missing_crosscheck)]
        raise ValueError(f"cross-check dates lack prior-day data: {missing}")
    requirements: dict[str, tuple[str, ...]] = {}
    btc_days = btc_targets | {day - timedelta(days=1) for day in btc_targets}
    requirements["BTCUSDT"] = tuple(day.isoformat() for day in sorted(btc_days))
    for symbol in CRYPTO_SYMBOLS[1:]:
        weekly_targets = set(
            eligible_dates(
                available.get(symbol, ()),
                TEST_START,
                TEST_END,
                weekly=True,
            )
        )
        weekly_days = weekly_targets | {
            day - timedelta(days=1) for day in weekly_targets
        }
        requirements[symbol] = tuple(
            day.isoformat() for day in sorted(weekly_days)
        )
    return requirements


def research_cache_dir(config: P2Config) -> Path:
    """Keep large derived streams beside an external-volume data lake by default."""
    if os.environ.get("P2_CACHE_DIR") or config.paths.lake_dir is None:
        return config.paths.cache_dir
    parts = config.paths.lake_dir.parts
    if len(parts) >= 3 and parts[:2] == ("/", "Volumes"):
        return Path("/Volumes") / parts[2] / "cache" / "p2"
    return config.paths.cache_dir


def ensure_bybit_cache(
    source_root: str | Path,
    cache_dir: str | Path,
    schedule: dict[str, Sequence[str]] | None = None,
    *,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, int]:
    """Reconstruct missing daily streams and return cache counts."""
    if workers < 1:
        raise ValueError("workers must be positive")
    requirements = schedule or required_bybit_dates(
        {
            symbol: available_bybit_dates(source_root, symbol)
            for symbol in CRYPTO_SYMBOLS
        }
    )
    destination = Path(cache_dir)
    required_count = 0
    cached_count = 0
    jobs: list[tuple[Path, Path, Path, str, str]] = []
    for symbol in CRYPTO_SYMBOLS:
        for date_value in requirements.get(symbol, ()):
            required_count += 1
            cache_path = destination / "bybit" / symbol / f"{date_value}.parquet"
            if cache_path.is_file() and cache_path.stat().st_size > 0:
                cached_count += 1
                continue
            orderbook_path, trades_path = find_bybit_files(
                source_root,
                symbol,
                date_value,
            )
            jobs.append(
                (
                    orderbook_path,
                    trades_path,
                    destination,
                    symbol,
                    date_value,
                )
            )
    if workers == 1:
        for job in jobs:
            _reconstruct_cache_job(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            tuple(executor.map(_reconstruct_cache_job, jobs))
    return {
        "required_days": required_count,
        "cached_days": cached_count,
        "reconstructed_days": len(jobs),
        "workers": workers,
    }


def _reconstruct_cache_job(job: tuple[Path, Path, Path, str, str]) -> None:
    orderbook_path, trades_path, destination, symbol, date_value = job
    reconstruct_bybit_day(
        orderbook_path,
        trades_path,
        destination,
        symbol,
        date_value,
    )


def run_research(
    config: P2Config,
    *,
    output_dir: str | Path | None = None,
) -> ResearchResult:
    """Execute studies B, A, C, and D and publish their fixed outputs."""
    if config.paths.lake_dir is None:
        raise ValueError("P2_LAKE_DIR must identify the full-history data root")
    started = perf_counter()
    destination = (
        Path(output_dir)
        if output_dir is not None
        else config.paths.results_dir / "published"
    )
    cache_dir = research_cache_dir(config)
    cache_summary = ensure_bybit_cache(
        config.paths.lake_dir,
        cache_dir,
    )
    queue = validate_es_queues(find_mbo_file(config.paths.lake_dir, "ES.c.0"))
    crypto = run_crypto_study(
        cache_dir,
        cancellation_rule=queue.selected_rule,
        workers=DEFAULT_WORKERS,
    )
    crosscheck = run_hft_crosscheck(
        cache_dir,
        crypto.daily_pnl,
        crypto.selection.strategies,
    )
    if config.paths.lobster_dir is None:
        raise ValueError("LOBSTER data directory is unavailable")
    lobster = run_lobster_appendix(
        config.paths.lobster_dir,
        crypto.selection.strategies,
        cancellation_rule=queue.selected_rule,
    )
    summary: dict[str, object] = {
        "runtime_seconds": perf_counter() - started,
        "cache": cache_summary,
        "queue_validation": {
            "dates": list(VALIDATION_DATES),
            "selected_rule": queue.selected_rule,
        },
        "crypto": crypto.summary,
        "external_crosscheck": {
            "dates": list(CROSSCHECK_DATES),
            "row_count": len(crosscheck),
        },
        "lobster_appendix": {
            "included_symbols": list(lobster.included_symbols),
            "missing_symbols": list(lobster.missing_symbols),
            "row_count": len(lobster.table),
        },
    }
    publish_research_outputs(
        destination,
        summary=summary,
        daily_pnl=crypto.daily_pnl,
        decomposition=crypto.decomposition,
        markouts=crypto.markouts,
        queue_validation=queue.table,
        hftbacktest_crosscheck=crosscheck,
        lobster_appendix=lobster.table,
    )
    return ResearchResult(output_dir=destination, summary=summary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_research(load_config(args.config), output_dir=args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
