"""End-to-end execution for the preregistered research study."""

from __future__ import annotations

import argparse
import json
import os
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
    run_crypto_study,
)


CRYPTO_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


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


def _weekly_test_dates() -> tuple[date, ...]:
    return tuple(day for day in _date_range(TEST_START, TEST_END) if day.weekday() == 2)


def required_bybit_dates() -> dict[str, tuple[str, ...]]:
    """Return the minimal raw-data schedule needed by all crypto studies."""
    btc_days = set(_date_range(SELECTION_START, TEST_END))
    btc_days.update(date.fromisoformat(value) for value in CROSSCHECK_DATES)
    btc_days.update(
        date.fromisoformat(value) - timedelta(days=1)
        for value in CROSSCHECK_DATES
    )
    weekly_days = set(_weekly_test_dates())
    weekly_days.update(day - timedelta(days=1) for day in tuple(weekly_days))
    return {
        "BTCUSDT": tuple(day.isoformat() for day in sorted(btc_days)),
        "ETHUSDT": tuple(day.isoformat() for day in sorted(weekly_days)),
        "SOLUSDT": tuple(day.isoformat() for day in sorted(weekly_days)),
    }


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
) -> dict[str, int]:
    """Reconstruct missing daily streams and return cache counts."""
    requirements = schedule or required_bybit_dates()
    destination = Path(cache_dir)
    required_count = 0
    cached_count = 0
    reconstructed_count = 0
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
            reconstruct_bybit_day(
                orderbook_path,
                trades_path,
                destination,
                symbol,
                date_value,
            )
            reconstructed_count += 1
    return {
        "required_days": required_count,
        "cached_days": cached_count,
        "reconstructed_days": reconstructed_count,
    }


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
