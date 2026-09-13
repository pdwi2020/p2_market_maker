"""Deterministic selection and crypto study execution."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from p2.bybit_replay import (
    CancellationRule,
    DailyReplayResult,
    ReplaySettings,
    StrategySpec,
    load_bybit_events,
    replay_bybit_day,
)
from p2.research_metrics import annualized_sharpe, summarize_performance
from p2.research_models import calibrate_bybit_day


SELECTION_START = date(2025, 5, 1)
SELECTION_END = date(2025, 6, 30)
TEST_START = date(2025, 7, 1)
TEST_END = date(2026, 4, 27)
GAMMAS = (0.00001, 0.00005, 0.0001, 0.0005, 0.001)
BETAS = (0.5, 1.0, 2.0, 4.0)
MAKER_FEE_RATES = (0.0002, 0.0001, 0.0, -0.00005)
LATENCIES_MS = (10, 50, 200)
PRIMARY_MAKER_FEE_RATE = 0.0002
PRIMARY_LATENCY_MS = 50

DAILY_COLUMNS = (
    "study",
    "symbol",
    "date",
    "calibration_date",
    "strategy",
    "maker_fee_rate",
    "taker_fee_rate",
    "latency_ms",
    "cancellation_rule",
    "gross_pnl",
    "fees",
    "net_pnl",
    "fill_count",
    "filled_volume",
    "quote_count",
    "quote_volume",
    "fill_ratio",
    "quote_to_fill",
    "mean_abs_inventory",
    "inventory_std",
    "max_abs_inventory",
    "end_inventory",
    "unobserved_placements",
)

DECOMPOSITION_COLUMNS = (
    "study",
    "symbol",
    "date",
    "strategy",
    "maker_fee_rate",
    "latency_ms",
    "realized_spread",
    "inventory_revaluation",
    "fee_cost",
    "net_pnl",
)

MARKOUT_COLUMNS = (
    "study",
    "symbol",
    "date",
    "strategy",
    "maker_fee_rate",
    "latency_ms",
    "horizon_seconds",
    "sample_count",
    "mean_markout",
)


@dataclass(frozen=True)
class SelectionResult:
    """Locked specifications and all selection-window trial statistics."""

    strategies: tuple[StrategySpec, ...]
    trial_sharpes: tuple[float, ...]
    daily_pnl: pd.DataFrame


@dataclass(frozen=True)
class CryptoStudyResult:
    """Study A tables and numeric summary data."""

    summary: dict[str, object]
    daily_pnl: pd.DataFrame
    decomposition: pd.DataFrame
    markouts: pd.DataFrame
    selection: SelectionResult


def strategy_candidates() -> tuple[StrategySpec, ...]:
    """Return the 31 preregistered candidates in stable order."""
    candidates = [StrategySpec("symmetric")]
    candidates.extend(StrategySpec("as", gamma=gamma) for gamma in GAMMAS)
    candidates.extend(StrategySpec("glft", gamma=gamma) for gamma in GAMMAS)
    candidates.extend(
        StrategySpec("glft_imbalance", gamma=gamma, beta=beta)
        for gamma in GAMMAS
        for beta in BETAS
    )
    return tuple(candidates)


def available_cache_dates(cache_dir: str | Path, symbol: str) -> tuple[date, ...]:
    """Discover reconstructed dates for one symbol."""
    directory = Path(cache_dir) / "bybit" / symbol
    dates: list[date] = []
    for path in directory.glob("*.parquet"):
        try:
            dates.append(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return tuple(sorted(set(dates)))


def eligible_dates(
    available: Iterable[date],
    start: date,
    end: date,
    *,
    weekly: bool = False,
) -> tuple[date, ...]:
    """Choose dates with a preceding-day calibration stream."""
    available_set = set(available)
    eligible = sorted(
        day
        for day in available_set
        if start <= day <= end and day - timedelta(days=1) in available_set
    )
    if not weekly:
        return tuple(eligible)
    selected: dict[tuple[int, int], date] = {}
    for day in eligible:
        iso = day.isocalendar()
        if day.weekday() < 2:
            continue
        selected.setdefault((iso.year, iso.week), day)
    return tuple(selected[key] for key in sorted(selected))


def _parameter_tuple(spec: StrategySpec) -> tuple[float, ...]:
    values = (spec.gamma, spec.beta)
    return tuple(float(value) for value in values if value is not None)


def _family(spec: StrategySpec) -> str:
    return spec.kind


def select_strategies(selection_daily: pd.DataFrame) -> SelectionResult:
    """Apply the locked family-wise Sharpe and tie-breaking rule."""
    candidates = strategy_candidates()
    stats: dict[str, tuple[float, float]] = {}
    for spec in candidates:
        values = selection_daily.loc[
            selection_daily["strategy"] == spec.name, "net_pnl"
        ].to_numpy(dtype=np.float64)
        if len(values) == 0:
            raise ValueError(f"selection results are missing {spec.name}")
        stats[spec.name] = (annualized_sharpe(values), float(np.mean(values)))

    selected: list[StrategySpec] = []
    for family in ("symmetric", "as", "glft", "glft_imbalance"):
        family_specs = [spec for spec in candidates if _family(spec) == family]

        def rank(spec: StrategySpec) -> tuple[float, float, tuple[float, ...]]:
            sharpe, mean = stats[spec.name]
            rounded = round(sharpe, 6) if np.isfinite(sharpe) else -np.inf
            return (-rounded, -mean, _parameter_tuple(spec))

        selected.append(min(family_specs, key=rank))
    trial_sharpes = tuple(stats[spec.name][0] for spec in candidates)
    return SelectionResult(
        strategies=tuple(selected),
        trial_sharpes=trial_sharpes,
        daily_pnl=selection_daily.copy(),
    )


def _stream_path(cache_dir: str | Path, symbol: str, day: date) -> Path:
    return Path(cache_dir) / "bybit" / symbol / f"{day.isoformat()}.parquet"


def run_selection(
    cache_dir: str | Path,
    *,
    cancellation_rule: CancellationRule,
    selection_dates: Sequence[date] | None = None,
    workers: int = 1,
) -> SelectionResult:
    """Replay all candidates on the selection window and lock each family."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if selection_dates is None:
        selection_dates = eligible_dates(
            available_cache_dates(cache_dir, "BTCUSDT"),
            SELECTION_START,
            SELECTION_END,
        )
    if not selection_dates:
        raise ValueError("no eligible BTCUSDT selection dates are available")
    tasks = [
        (Path(cache_dir), cancellation_rule, trading_day)
        for trading_day in selection_dates
    ]
    if workers == 1:
        daily_rows = [_selection_day_rows(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            daily_rows = list(executor.map(_selection_day_rows, tasks))
    rows = [row for day_rows in daily_rows for row in day_rows]
    return select_strategies(pd.DataFrame(rows))


def _selection_day_rows(
    task: tuple[Path, CancellationRule, date],
) -> list[dict[str, object]]:
    cache_dir, cancellation_rule, trading_day = task
    calibration_day = trading_day - timedelta(days=1)
    calibration = calibrate_bybit_day(
        _stream_path(cache_dir, "BTCUSDT", calibration_day)
    )
    events = load_bybit_events(_stream_path(cache_dir, "BTCUSDT", trading_day))
    settings = ReplaySettings(
        maker_fee_rate=PRIMARY_MAKER_FEE_RATE,
        latency_ms=PRIMARY_LATENCY_MS,
        cancellation_rule=cancellation_rule,
    )
    rows: list[dict[str, object]] = []
    for spec in strategy_candidates():
        result = replay_bybit_day(
            events,
            symbol="BTCUSDT",
            date=trading_day.isoformat(),
            strategy=spec,
            calibration=calibration,
            settings=settings,
        )
        rows.append(
            {
                "date": trading_day.isoformat(),
                "strategy": spec.name,
                "net_pnl": result.net_pnl,
            }
        )
    return rows


def replay_result_rows(
    result: DailyReplayResult,
    *,
    calibration_date: str,
    settings: ReplaySettings,
    maker_fee_rates: Sequence[float] = MAKER_FEE_RATES,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Expand one fee-neutral replay into the fixed maker-rate grid."""
    notional = float(sum(fill.price * fill.size for fill in result.fills))
    daily_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    markout_rows: list[dict[str, object]] = []
    for maker_fee_rate in maker_fee_rates:
        fees = maker_fee_rate * notional
        net_pnl = result.gross_pnl - fees
        common = {
            "study": "crypto",
            "symbol": result.symbol,
            "date": result.date,
            "strategy": result.strategy,
            "maker_fee_rate": maker_fee_rate,
            "latency_ms": settings.latency_ms,
        }
        daily_rows.append(
            {
                **common,
                "calibration_date": calibration_date,
                "taker_fee_rate": settings.taker_fee_rate,
                "cancellation_rule": settings.cancellation_rule,
                "gross_pnl": result.gross_pnl,
                "fees": fees,
                "net_pnl": net_pnl,
                "fill_count": result.fill_count,
                "filled_volume": result.filled_volume,
                "quote_count": result.quote_count,
                "quote_volume": result.quote_volume,
                "fill_ratio": result.fill_ratio,
                "quote_to_fill": result.quote_to_fill,
                "mean_abs_inventory": result.mean_abs_inventory,
                "inventory_std": result.inventory_std,
                "max_abs_inventory": result.max_abs_inventory,
                "end_inventory": result.end_inventory,
                "unobserved_placements": result.unobserved_placements,
            }
        )
        decomposition_rows.append(
            {
                **common,
                "realized_spread": result.decomposition.realized_spread,
                "inventory_revaluation": result.decomposition.inventory_revaluation,
                "fee_cost": fees,
                "net_pnl": net_pnl,
            }
        )
        for horizon, values in sorted(result.markouts.items()):
            markout_rows.append(
                {
                    **common,
                    "horizon_seconds": horizon,
                    "sample_count": len(values),
                    "mean_markout": float(np.mean(values)) if values else np.nan,
                }
            )
    return daily_rows, decomposition_rows, markout_rows


def _main_table(
    daily_pnl: pd.DataFrame,
    trial_sharpes: Sequence[float],
    *,
    bootstrap_resamples: int,
) -> list[dict[str, object]]:
    primary = daily_pnl[
        np.isclose(daily_pnl["maker_fee_rate"], PRIMARY_MAKER_FEE_RATE)
        & (daily_pnl["latency_ms"] == PRIMARY_LATENCY_MS)
    ]
    table: list[dict[str, object]] = []
    for (symbol, strategy), rows in primary.groupby(["symbol", "strategy"], sort=True):
        rows = rows.sort_values("date")
        performance = summarize_performance(
            rows["net_pnl"].to_numpy(),
            trial_sharpes,
            bootstrap_resamples=bootstrap_resamples,
        )
        fill_count = int(rows["fill_count"].sum())
        quote_volume = float(rows["quote_volume"].sum())
        table.append(
            {
                "symbol": symbol,
                "strategy": strategy,
                **asdict(performance),
                "gross_pnl": float(rows["gross_pnl"].sum()),
                "fees": float(rows["fees"].sum()),
                "fill_count": fill_count,
                "filled_volume": float(rows["filled_volume"].sum()),
                "fill_ratio": (
                    float(rows["filled_volume"].sum()) / quote_volume
                    if quote_volume
                    else np.nan
                ),
                "quote_to_fill": (
                    float(rows["quote_count"].sum()) / fill_count
                    if fill_count
                    else np.nan
                ),
                "mean_abs_inventory": float(rows["mean_abs_inventory"].mean()),
                "inventory_std": float(rows["inventory_std"].mean()),
                "max_abs_inventory": float(rows["max_abs_inventory"].max()),
                "mean_end_inventory": float(rows["end_inventory"].mean()),
            }
        )
    return table


def run_crypto_study(
    cache_dir: str | Path,
    *,
    cancellation_rule: CancellationRule,
    selection_dates: Sequence[date] | None = None,
    test_dates: dict[str, Sequence[date]] | None = None,
    bootstrap_resamples: int = 10_000,
    workers: int = 1,
) -> CryptoStudyResult:
    """Run selection and all Study A primary and sensitivity cells."""
    if workers < 1:
        raise ValueError("workers must be positive")
    started = perf_counter()
    selection = run_selection(
        cache_dir,
        cancellation_rule=cancellation_rule,
        selection_dates=selection_dates,
        workers=workers,
    )
    if test_dates is None:
        test_dates = {
            symbol: eligible_dates(
                available_cache_dates(cache_dir, symbol),
                TEST_START,
                TEST_END,
                weekly=symbol != "BTCUSDT",
            )
            for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
        }
    tasks = [
        (
            Path(cache_dir),
            symbol,
            trading_day,
            selection.strategies,
            cancellation_rule,
        )
        for symbol, dates in test_dates.items()
        for trading_day in dates
    ]
    if workers == 1:
        result_rows = [_test_day_rows(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            result_rows = list(executor.map(_test_day_rows, tasks))
    daily_rows = [row for result in result_rows for row in result[0]]
    decomposition_rows = [row for result in result_rows for row in result[1]]
    markout_rows = [row for result in result_rows for row in result[2]]
    daily_pnl = pd.DataFrame(daily_rows, columns=DAILY_COLUMNS)
    decomposition = pd.DataFrame(decomposition_rows, columns=DECOMPOSITION_COLUMNS)
    markouts = pd.DataFrame(markout_rows, columns=MARKOUT_COLUMNS)
    if daily_pnl.empty:
        raise ValueError("no eligible crypto test dates are available")
    main_table = _main_table(
        daily_pnl,
        selection.trial_sharpes,
        bootstrap_resamples=bootstrap_resamples,
    )
    summary: dict[str, object] = {
        "selection_trial_count": len(strategy_candidates()),
        "selection_dates": sorted(selection.daily_pnl["date"].unique().tolist()),
        "selection_trial_sharpes": [
            {"strategy": spec.name, "annualized_sharpe": sharpe}
            for spec, sharpe in zip(
                strategy_candidates(), selection.trial_sharpes, strict=True
            )
        ],
        "selected_strategies": [asdict(spec) for spec in selection.strategies],
        "main_study": main_table,
        "runtime_seconds": perf_counter() - started,
    }
    return CryptoStudyResult(
        summary=summary,
        daily_pnl=daily_pnl,
        decomposition=decomposition,
        markouts=markouts,
        selection=selection,
    )


def _test_day_rows(
    task: tuple[
        Path,
        str,
        date,
        tuple[StrategySpec, ...],
        CancellationRule,
    ],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    cache_dir, symbol, trading_day, strategies, cancellation_rule = task
    calibration_day = trading_day - timedelta(days=1)
    calibration = calibrate_bybit_day(
        _stream_path(cache_dir, symbol, calibration_day)
    )
    events = load_bybit_events(_stream_path(cache_dir, symbol, trading_day))
    daily_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    markout_rows: list[dict[str, object]] = []
    for spec in strategies:
        for latency_ms in LATENCIES_MS:
            settings = ReplaySettings(
                maker_fee_rate=0.0,
                latency_ms=latency_ms,
                cancellation_rule=cancellation_rule,
            )
            result = replay_bybit_day(
                events,
                symbol=symbol,
                date=trading_day.isoformat(),
                strategy=spec,
                calibration=calibration,
                settings=settings,
            )
            daily, decomposition, markouts = replay_result_rows(
                result,
                calibration_date=calibration_day.isoformat(),
                settings=settings,
            )
            daily_rows.extend(daily)
            decomposition_rows.extend(decomposition)
            markout_rows.extend(markouts)
    return daily_rows, decomposition_rows, markout_rows
