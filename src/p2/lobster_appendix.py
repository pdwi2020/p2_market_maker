"""Locked cross-symbol LOBSTER appendix execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from p2.bybit_replay import CancellationRule, StrategySpec
from p2.calibration import calibrate_arrival_params, calibrate_sigma
from p2.cross_symbol_sweep import ensure_lobster_data
from p2.lobster_replay import BacktestResult, LOBSTERReplayer
from p2.research_models import (
    apply_imbalance_skew,
    avellaneda_stoikov_quotes,
    glft_quotes,
    symmetric_touch_quotes,
)


LOBSTER_SYMBOLS = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")
LOBSTER_DATE = "2012-06-21"
LOBSTER_LEVEL = 10
CALIBRATION_SECONDS = 1_800.0
TICK_SIZE = 0.01
ORDER_SIZE = 1
INVENTORY_LIMIT = 10
REQUOTE_MS = 100.0
LATENCY_MS = 1.0
MAKER_REBATE_PER_SHARE = 0.0025
TAKER_FEE_PER_SHARE = 0.0030

LOBSTER_APPENDIX_COLUMNS = (
    "study",
    "symbol",
    "date",
    "strategy",
    "calibration_event_count",
    "replay_event_count",
    "calibration_seconds",
    "replay_seconds",
    "sigma",
    "A",
    "kappa",
    "gamma",
    "beta",
    "tick_size",
    "order_size",
    "inventory_limit",
    "requote_ms",
    "latency_ms",
    "cancellation_rule",
    "gross_pnl",
    "maker_rebates_pnl",
    "taker_fees_pnl",
    "net_pnl",
    "fill_count",
    "filled_volume",
    "quoted_width",
    "realized_spread",
    "spread_capture_pct",
    "realized_spread_pnl",
    "inventory_mtm_pnl",
    "mean_abs_inventory",
    "inventory_std",
    "max_abs_inventory",
    "end_inventory",
    "markout_1s",
    "markout_5s",
    "markout_30s",
    "markout_60s",
    "avg_position_at_fill",
    "cancellation_rate",
    "time_to_fill_mean",
    "time_to_fill_p50",
    "time_to_fill_p90",
    "marketable_fill_count",
)


@dataclass(frozen=True)
class LobsterAppendixResult:
    """Appendix table plus explicit data-availability accounting."""

    table: pd.DataFrame
    included_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]


TouchStrategy = Callable[
    [float, int, float, float, float, float, float],
    tuple[float, float],
]


def split_calibration_replay(
    replayer: LOBSTERReplayer,
    calibration_seconds: float = CALIBRATION_SECONDS,
) -> tuple[LOBSTERReplayer, LOBSTERReplayer]:
    """Split one session into an initial calibration and later replay window."""
    if replayer.orderbook is None or replayer.messages is None:
        raise RuntimeError("LOBSTER data not loaded. Call load(...) first.")
    if calibration_seconds <= 0.0:
        raise ValueError("calibration_seconds must be positive")
    n_rows = min(len(replayer.orderbook), len(replayer.messages))
    if n_rows == 0:
        raise ValueError("LOBSTER session is empty")
    orderbook = replayer.orderbook.iloc[:n_rows]
    messages = replayer.messages.iloc[:n_rows]
    times = messages["time"].to_numpy(dtype=float)
    cutoff = float(times[0]) + calibration_seconds
    split_idx = int(np.searchsorted(times, cutoff, side="left"))
    if split_idx == 0 or split_idx == n_rows:
        raise ValueError("LOBSTER session does not span both study windows")

    calibration = LOBSTERReplayer()
    calibration.orderbook = orderbook.iloc[:split_idx].reset_index(drop=True)
    calibration.messages = messages.iloc[:split_idx].reset_index(drop=True)
    replay = LOBSTERReplayer()
    replay.orderbook = orderbook.iloc[split_idx:].reset_index(drop=True)
    replay.messages = messages.iloc[split_idx:].reset_index(drop=True)
    return calibration, replay


def lobster_strategy(
    spec: StrategySpec,
    *,
    sigma: float,
    A: float,
    kappa: float,
    horizon_seconds: float,
    tick_size: float = TICK_SIZE,
) -> TouchStrategy:
    """Create one touch-aware strategy using the selected crypto parameters."""
    if min(sigma, A, kappa, horizon_seconds, tick_size) <= 0.0:
        raise ValueError("calibration, horizon, and tick values must be positive")

    def quote(
        mid: float,
        inventory: int,
        elapsed_seconds: float,
        best_bid: float,
        best_ask: float,
        bid_size: float,
        ask_size: float,
    ) -> tuple[float, float]:
        if spec.kind == "symmetric":
            bid, ask = symmetric_touch_quotes(best_bid, best_ask)
        elif spec.kind == "as":
            if spec.gamma is None:
                raise ValueError("AS strategy requires gamma")
            bid, ask = avellaneda_stoikov_quotes(
                mid,
                float(inventory),
                elapsed_seconds,
                gamma=spec.gamma,
                sigma=sigma,
                kappa=kappa,
                horizon_seconds=horizon_seconds,
            )
        else:
            if spec.gamma is None:
                raise ValueError("GLFT strategy requires gamma")
            bid, ask = glft_quotes(
                mid,
                float(inventory),
                gamma=spec.gamma,
                sigma=sigma,
                A=A,
                kappa=kappa,
                order_size=float(ORDER_SIZE),
            )
            if spec.kind == "glft_imbalance":
                if spec.beta is None:
                    raise ValueError("imbalance strategy requires beta")
                total_size = bid_size + ask_size
                imbalance = (bid_size - ask_size) / total_size if total_size else 0.0
                bid, ask = apply_imbalance_skew(
                    bid,
                    ask,
                    float(imbalance),
                    beta=spec.beta,
                    tick_size=tick_size,
                )
        rounded_bid = np.floor((bid + 1e-12) / tick_size) * tick_size
        rounded_ask = np.ceil((ask - 1e-12) / tick_size) * tick_size
        return float(rounded_bid), float(rounded_ask)

    return quote


def _event_horizon(replayer: LOBSTERReplayer) -> float:
    if replayer.messages is None or replayer.messages.empty:
        return 0.0
    times = replayer.messages["time"].to_numpy(dtype=float)
    return float(max(times[-1] - times[0], 0.0))


def _result_row(
    *,
    symbol: str,
    spec: StrategySpec,
    result: BacktestResult,
    calibration_event_count: int,
    replay_event_count: int,
    replay_seconds: float,
    sigma: float,
    A: float,
    kappa: float,
    cancellation_rule: CancellationRule,
) -> dict[str, object]:
    inventory = np.asarray(result.inventory_path, dtype=float)
    gross_pnl = result.pnl - result.fees_and_rebates_pnl
    return {
        "study": "lobster_appendix",
        "symbol": symbol,
        "date": LOBSTER_DATE,
        "strategy": spec.name,
        "calibration_event_count": calibration_event_count,
        "replay_event_count": replay_event_count,
        "calibration_seconds": CALIBRATION_SECONDS,
        "replay_seconds": replay_seconds,
        "sigma": sigma,
        "A": A,
        "kappa": kappa,
        "gamma": spec.gamma,
        "beta": spec.beta,
        "tick_size": TICK_SIZE,
        "order_size": ORDER_SIZE,
        "inventory_limit": INVENTORY_LIMIT,
        "requote_ms": REQUOTE_MS,
        "latency_ms": LATENCY_MS,
        "cancellation_rule": cancellation_rule,
        "gross_pnl": gross_pnl,
        "maker_rebates_pnl": result.maker_rebates_pnl,
        "taker_fees_pnl": result.taker_fees_pnl,
        "net_pnl": result.pnl,
        "fill_count": len(result.fills),
        "filled_volume": result.fill_quantity,
        "quoted_width": result.quoted_width,
        "realized_spread": result.realized_spread,
        "spread_capture_pct": result.spread_capture_pct,
        "realized_spread_pnl": result.realized_spread_pnl,
        "inventory_mtm_pnl": result.inventory_mtm_pnl,
        "mean_abs_inventory": float(np.mean(np.abs(inventory))),
        "inventory_std": float(np.std(inventory)),
        "max_abs_inventory": float(np.max(np.abs(inventory))),
        "end_inventory": int(inventory[-1]),
        "markout_1s": result.markout_1s,
        "markout_5s": result.markout_5s,
        "markout_30s": result.markout_30s,
        "markout_60s": result.markout_60s,
        "avg_position_at_fill": result.avg_position_at_fill,
        "cancellation_rate": result.cancellation_rate,
        "time_to_fill_mean": result.time_to_fill_mean,
        "time_to_fill_p50": result.time_to_fill_p50,
        "time_to_fill_p90": result.time_to_fill_p90,
        "marketable_fill_count": result.marketable_fill_count,
    }


def run_lobster_appendix(
    data_dir: str | Path,
    selected_strategies: Sequence[StrategySpec],
    *,
    cancellation_rule: CancellationRule,
    symbols: Sequence[str] = LOBSTER_SYMBOLS,
) -> LobsterAppendixResult:
    """Run the fixed sample-day appendix for every locally available symbol."""
    if not selected_strategies:
        raise ValueError("selected_strategies must not be empty")
    rows: list[dict[str, object]] = []
    included: list[str] = []
    missing: list[str] = []
    for raw_symbol in symbols:
        symbol = raw_symbol.upper()
        try:
            message_path, orderbook_path = ensure_lobster_data(
                symbol,
                LOBSTER_DATE,
                LOBSTER_LEVEL,
                Path(data_dir),
                allow_download=False,
            )
        except FileNotFoundError:
            missing.append(symbol)
            continue

        loaded = LOBSTERReplayer().load(orderbook_path, message_path)
        calibration, replay = split_calibration_replay(loaded)
        if calibration.orderbook is None or calibration.messages is None:
            raise RuntimeError("calibration split is unavailable")
        if replay.messages is None:
            raise RuntimeError("replay split is unavailable")
        sigma = calibrate_sigma(calibration.orderbook, calibration.messages)
        A, kappa = calibrate_arrival_params(
            calibration.orderbook,
            calibration.messages,
        )
        replay_seconds = _event_horizon(replay)
        if min(sigma, A, kappa, replay_seconds) <= 0.0:
            raise ValueError(f"invalid LOBSTER calibration for {symbol}")
        included.append(symbol)
        for spec in selected_strategies:
            result = replay.run_strategy(
                lobster_strategy(
                    spec,
                    sigma=sigma,
                    A=A,
                    kappa=kappa,
                    horizon_seconds=replay_seconds,
                ),
                use_queue_position=True,
                session_duration_seconds=replay_seconds,
                post_only_mode="reprice",
                order_size=ORDER_SIZE,
                cancellation_rule=cancellation_rule,
                latency_ms=LATENCY_MS,
                requote_ms=REQUOTE_MS,
                touch_aware=True,
                inventory_limit=INVENTORY_LIMIT,
                maker_rebate_per_share=MAKER_REBATE_PER_SHARE,
                taker_fee_per_share=TAKER_FEE_PER_SHARE,
            )
            rows.append(
                _result_row(
                    symbol=symbol,
                    spec=spec,
                    result=result,
                    calibration_event_count=len(calibration.messages),
                    replay_event_count=len(replay.messages),
                    replay_seconds=replay_seconds,
                    sigma=sigma,
                    A=A,
                    kappa=kappa,
                    cancellation_rule=cancellation_rule,
                )
            )
    return LobsterAppendixResult(
        table=pd.DataFrame(rows, columns=LOBSTER_APPENDIX_COLUMNS),
        included_symbols=tuple(included),
        missing_symbols=tuple(missing),
    )
