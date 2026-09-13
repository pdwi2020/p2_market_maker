"""Independent replay cross-check through the external hftbacktest package."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
import pandas as pd

from p2.bybit_replay import BybitEventArrays, StrategySpec, load_bybit_events
from p2.research_models import DailyCalibration, calibrate_bybit_day, glft_quotes
from p2.research_study import PRIMARY_LATENCY_MS, PRIMARY_MAKER_FEE_RATE


CROSSCHECK_DATES = (
    "2025-07-07",
    "2025-08-04",
    "2025-10-06",
    "2026-01-05",
    "2026-04-06",
)


@dataclass(frozen=True)
class ExternalReplayResult:
    """Aggregate outputs reported by one external-engine replay."""

    fill_count: int
    filled_volume: float
    gross_pnl: float
    fees: float
    net_pnl: float
    end_inventory: float


def _bindings() -> ModuleType:
    try:
        return importlib.import_module("hftbacktest")
    except ImportError as exc:
        raise RuntimeError(
            "hftbacktest is required for the external replay cross-check"
        ) from exc


def _event_levels(
    offsets: np.ndarray,
    prices: np.ndarray,
    sizes: np.ndarray,
    row: int,
) -> dict[float, float]:
    start, end = int(offsets[row]), int(offsets[row + 1])
    return {
        float(price): float(size)
        for price, size in zip(prices[start:end], sizes[start:end], strict=True)
        if np.isfinite(price) and np.isfinite(size) and size > 0.0
    }


def normalize_bybit_events(
    events: BybitEventArrays,
    bindings: ModuleType | None = None,
) -> np.ndarray:
    """Convert a reconstructed stream to the package's normalized event array."""
    hft = bindings or _bindings()
    capacity = max(len(events.times) * 4, 64)
    output = np.zeros(capacity, dtype=hft.event_dtype)
    count = 0
    common_flags = hft.EXCH_EVENT | hft.LOCAL_EVENT

    def append(event_type: int, timestamp_ms: int, price: float, size: float) -> None:
        nonlocal output, count
        if count == len(output):
            expanded = np.zeros(len(output) * 2, dtype=hft.event_dtype)
            expanded[:count] = output
            output = expanded
        output[count]["ev"] = event_type | common_flags
        timestamp_ns = int(timestamp_ms) * 1_000_000
        output[count]["exch_ts"] = timestamp_ns
        output[count]["local_ts"] = timestamp_ns
        output[count]["px"] = price
        output[count]["qty"] = size
        count += 1

    previous_bids: dict[float, float] = {}
    previous_asks: dict[float, float] = {}
    book_row = 0
    trade_row = 0
    initialized = False
    for event_row, timestamp_ms in enumerate(events.times):
        if events.is_book[event_row]:
            if events.book_valid[book_row]:
                bids = _event_levels(
                    events.bid_offsets,
                    events.bid_prices,
                    events.bid_sizes,
                    book_row,
                )
                asks = _event_levels(
                    events.ask_offsets,
                    events.ask_prices,
                    events.ask_sizes,
                    book_row,
                )
            else:
                bids = {}
                asks = {}
            for side_flag, previous, current in (
                (hft.BUY_EVENT, previous_bids, bids),
                (hft.SELL_EVENT, previous_asks, asks),
            ):
                for price in previous.keys() - current.keys():
                    append(hft.DEPTH_EVENT | side_flag, timestamp_ms, price, 0.0)
                depth_event = hft.DEPTH_EVENT if initialized else hft.DEPTH_SNAPSHOT_EVENT
                for price, size in current.items():
                    if previous.get(price) != size:
                        append(depth_event | side_flag, timestamp_ms, price, size)
            previous_bids, previous_asks = bids, asks
            initialized = bool(bids and asks)
            book_row += 1
        else:
            side = str(events.trade_sides[trade_row]).lower()
            side_flag = hft.BUY_EVENT if side == "buy" else hft.SELL_EVENT
            append(
                hft.TRADE_EVENT | side_flag,
                timestamp_ms,
                float(events.trade_prices[trade_row]),
                float(events.trade_volumes[trade_row]),
            )
            trade_row += 1
    return output[:count].copy()


def _strategy_prices(
    spec: StrategySpec,
    calibration: DailyCalibration,
    *,
    best_bid: float,
    best_ask: float,
    inventory: float,
) -> tuple[float, float]:
    if spec.kind == "symmetric":
        return best_bid, best_ask
    if spec.kind != "glft" or spec.gamma is None:
        raise ValueError("external cross-check supports symmetric and GLFT")
    return glft_quotes(
        0.5 * (best_bid + best_ask),
        inventory,
        gamma=spec.gamma,
        sigma=calibration.sigma,
        A=calibration.A,
        kappa=calibration.kappa,
        order_size=0.01,
    )


def _manage_orders(
    hbt: object,
    hft: ModuleType,
    *,
    bid_tick: int | None,
    ask_tick: int | None,
    next_order_id: int,
) -> int:
    orders = hbt.orders(0)
    update_bid = bid_tick is not None
    update_ask = ask_tick is not None
    values = orders.values()
    while values.has_next():
        order = values.get()
        if order.side == hft.BUY:
            same = bid_tick is not None and order.price_tick == bid_tick
            update_bid &= not same
            if order.cancellable and not same:
                hbt.cancel(0, order.order_id, False)
            elif not order.cancellable and not same:
                update_bid = False
        elif order.side == hft.SELL:
            same = ask_tick is not None and order.price_tick == ask_tick
            update_ask &= not same
            if order.cancellable and not same:
                hbt.cancel(0, order.order_id, False)
            elif not order.cancellable and not same:
                update_ask = False
    if update_bid and bid_tick is not None:
        hbt.submit_buy_order(
            0,
            next_order_id,
            bid_tick * 0.1,
            0.01,
            hft.GTX,
            hft.LIMIT,
            False,
        )
        next_order_id += 1
    if update_ask and ask_tick is not None:
        hbt.submit_sell_order(
            0,
            next_order_id,
            ask_tick * 0.1,
            0.01,
            hft.GTX,
            hft.LIMIT,
            False,
        )
        next_order_id += 1
    return next_order_id


def run_external_replay(
    normalized_events: np.ndarray,
    *,
    strategy: StrategySpec,
    calibration: DailyCalibration,
    latency_ms: int = PRIMARY_LATENCY_MS,
    maker_fee_rate: float = PRIMARY_MAKER_FEE_RATE,
) -> ExternalReplayResult:
    """Run one strategy through hftbacktest's independent replay engine."""
    hft = _bindings()
    asset = (
        hft.BacktestAsset()
        .data(normalized_events)
        .linear_asset(1.0)
        .constant_order_latency(latency_ms * 1_000_000, latency_ms * 1_000_000)
        .risk_adverse_queue_model()
        .partial_fill_exchange()
        .trading_value_fee_model(maker_fee_rate, 0.00055)
        .tick_size(0.1)
        .lot_size(0.01)
    )
    hbt = hft.HashMapMarketDepthBacktest([asset])
    next_order_id = 1
    try:
        while hbt.elapse(100_000_000) == 0:
            hbt.clear_inactive_orders(0)
            depth = hbt.depth(0)
            if not np.isfinite(depth.best_bid) or not np.isfinite(depth.best_ask):
                continue
            inventory = float(hbt.position(0))
            bid, ask = _strategy_prices(
                strategy,
                calibration,
                best_bid=float(depth.best_bid),
                best_ask=float(depth.best_ask),
                inventory=inventory,
            )
            bid_tick = int(np.floor(bid / 0.1 + 1e-9))
            ask_tick = int(np.ceil(ask / 0.1 - 1e-9))
            if inventory + 0.01 > 0.05 + 1e-12:
                bid_tick = None
            if inventory - 0.01 < -0.05 - 1e-12:
                ask_tick = None
            next_order_id = _manage_orders(
                hbt,
                hft,
                bid_tick=bid_tick,
                ask_tick=ask_tick,
                next_order_id=next_order_id,
            )
        depth = hbt.depth(0)
        midpoint = 0.5 * (float(depth.best_bid) + float(depth.best_ask))
        state = hbt.state_values(0)
        gross_pnl = float(state.balance + state.position * midpoint)
        return ExternalReplayResult(
            fill_count=int(state.num_trades),
            filled_volume=float(state.trading_volume),
            gross_pnl=gross_pnl,
            fees=float(state.fee),
            net_pnl=gross_pnl - float(state.fee),
            end_inventory=float(state.position),
        )
    finally:
        hbt.close()


def _relative_difference(external: float, native: float) -> float:
    return (external - native) / abs(native) if native != 0.0 else np.nan


def comparison_row(
    *,
    trading_date: str,
    strategy: str,
    native: dict[str, float | int],
    external: ExternalReplayResult,
    package_version: str,
) -> dict[str, object]:
    """Build one published absolute and relative comparison row."""
    row: dict[str, object] = {
        "study": "external_crosscheck",
        "symbol": "BTCUSDT",
        "date": trading_date,
        "strategy": strategy,
        "package_version": package_version,
    }
    for metric in ("fill_count", "filled_volume", "net_pnl"):
        native_value = float(native[metric])
        external_value = float(getattr(external, metric))
        row[f"native_{metric}"] = native_value
        row[f"external_{metric}"] = external_value
        row[f"absolute_difference_{metric}"] = external_value - native_value
        row[f"relative_difference_{metric}"] = _relative_difference(
            external_value, native_value
        )
    return row


def run_hft_crosscheck(
    cache_dir: str | Path,
    native_daily: pd.DataFrame,
    selected_strategies: Sequence[StrategySpec],
    *,
    dates: Sequence[str] = CROSSCHECK_DATES,
) -> pd.DataFrame:
    """Run the two locked strategies on all five cross-check dates."""
    hft = _bindings()
    symmetric = next(spec for spec in selected_strategies if spec.kind == "symmetric")
    selected_glft = next(spec for spec in selected_strategies if spec.kind == "glft")
    rows: list[dict[str, object]] = []
    for date_value in dates:
        trading_day = date.fromisoformat(date_value)
        calibration_day = trading_day - timedelta(days=1)
        trading_path = Path(cache_dir) / "bybit" / "BTCUSDT" / f"{date_value}.parquet"
        calibration_path = (
            Path(cache_dir)
            / "bybit"
            / "BTCUSDT"
            / f"{calibration_day.isoformat()}.parquet"
        )
        calibration = calibrate_bybit_day(calibration_path)
        normalized = normalize_bybit_events(load_bybit_events(trading_path), hft)
        for strategy in (symmetric, selected_glft):
            external = run_external_replay(
                normalized,
                strategy=strategy,
                calibration=calibration,
            )
            match = native_daily[
                (native_daily["date"] == date_value)
                & (native_daily["strategy"] == strategy.name)
                & np.isclose(
                    native_daily["maker_fee_rate"], PRIMARY_MAKER_FEE_RATE
                )
                & (native_daily["latency_ms"] == PRIMARY_LATENCY_MS)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"native primary result is missing for {date_value} {strategy.name}"
                )
            rows.append(
                comparison_row(
                    trading_date=date_value,
                    strategy=strategy.name,
                    native=match.iloc[0].to_dict(),
                    external=external,
                    package_version=hft.__version__,
                )
            )
    return pd.DataFrame(rows)
