"""Simplified top-of-book replay scaffold for LOBSTER-style CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd


DEFAULT_SESSION_DURATION_SECONDS = 23_400.0


@dataclass(slots=True)
class BacktestResult:
    pnl: float
    inventory_path: np.ndarray
    fill_times: list[float]
    spread_realized: float
    avg_position_at_fill: float = 0.0
    cancellation_rate: float = 0.0
    time_to_fill_mean: float = 0.0
    time_to_fill_p50: float = 0.0
    time_to_fill_p90: float = 0.0
    marketable_fill_count: int = 0


@dataclass(slots=True)
class _RestingOrder:
    price: float | None = None
    position: int | None = None
    remaining_size: int = 0
    placed_at: float | None = None


class LOBSTERReplayer:
    """Replay top-of-book LOBSTER data with a deliberately simplified fill model.

    This is not a queue-position simulator. It uses event direction and trade price
    against generated quotes to approximate whether our quote would have traded.
    """

    def __init__(self) -> None:
        self.orderbook: pd.DataFrame | None = None
        self.messages: pd.DataFrame | None = None

    def load(self, orderbook_file: str | Path, message_file: str | Path) -> "LOBSTERReplayer":
        orderbook_path = Path(orderbook_file)
        message_path = Path(message_file)

        if not orderbook_path.exists():
            raise FileNotFoundError(f"LOBSTER orderbook file not found: {orderbook_path}")
        if not message_path.exists():
            raise FileNotFoundError(f"LOBSTER message file not found: {message_path}")

        self.orderbook = self._label_orderbook(pd.read_csv(orderbook_path, header=None))
        self.messages = self._label_messages(pd.read_csv(message_path, header=None))
        self._normalize_price_scale()
        return self

    @staticmethod
    def _label_orderbook(frame: pd.DataFrame) -> pd.DataFrame:
        names: list[str] = []
        for level in range(1, frame.shape[1] // 4 + 1):
            names.extend(
                [
                    f"ask_price_{level}",
                    f"ask_size_{level}",
                    f"bid_price_{level}",
                    f"bid_size_{level}",
                ]
            )
        if len(names) < frame.shape[1]:
            names.extend([f"col_{idx}" for idx in range(len(names), frame.shape[1])])
        labeled = frame.copy()
        labeled.columns = names[: frame.shape[1]]
        return labeled

    @staticmethod
    def _label_messages(frame: pd.DataFrame) -> pd.DataFrame:
        names = ["time", "event_type", "order_id", "size", "price", "direction"]
        if frame.shape[1] > 6:
            names.extend([f"col_{idx}" for idx in range(6, frame.shape[1])])
        labeled = frame.copy()
        labeled.columns = names[: frame.shape[1]]
        return labeled

    def _normalize_price_scale(self) -> None:
        if self.orderbook is None or self.messages is None:
            return
        if float(self.orderbook["ask_price_1"].median()) > 10_000:
            for column in [col for col in self.orderbook.columns if "price" in col]:
                self.orderbook[column] = self.orderbook[column] / 10_000.0
            if "price" in self.messages.columns:
                self.messages["price"] = self.messages["price"] / 10_000.0

    def mid_price_series(self) -> pd.Series:
        if self.orderbook is None:
            raise RuntimeError("Orderbook not loaded. Call load(...) first.")
        return 0.5 * (self.orderbook["ask_price_1"] + self.orderbook["bid_price_1"])

    def run_strategy(
        self,
        strategy_fn: Callable[[float, int, float], tuple[float, float]],
        *,
        use_queue_position: bool = False,
        session_duration_seconds: float = DEFAULT_SESSION_DURATION_SECONDS,
        post_only_mode: Literal["reprice", "reject"] = "reprice",
        order_size: int = 1,
        cancellation_rule: Literal["cancel-from-back", "proportional"] = "proportional",
        latency_ms: float = 1.0,
    ) -> BacktestResult:
        if latency_ms < 0.0:
            raise ValueError("latency_ms must be non-negative")
        if use_queue_position:
            return self._run_queue_aware_strategy(
                strategy_fn,
                session_duration_seconds,
                post_only_mode,
                order_size,
                cancellation_rule,
                latency_ms,
            )
        return self._run_instantaneous_strategy(
            strategy_fn,
            session_duration_seconds,
            post_only_mode,
            latency_ms,
        )

    def _run_instantaneous_strategy(
        self,
        strategy_fn: Callable[[float, int, float], tuple[float, float]],
        session_duration_seconds: float,
        post_only_mode: Literal["reprice", "reject"],
        latency_ms: float,
    ) -> BacktestResult:
        if self.orderbook is None or self.messages is None:
            raise RuntimeError("LOBSTER data not loaded. Call load(...) first.")

        bid_price_post = self.orderbook["bid_price_1"].to_numpy(dtype=float)
        ask_price_post = self.orderbook["ask_price_1"].to_numpy(dtype=float)
        bid_price_pre = np.concatenate(([bid_price_post[0]], bid_price_post[:-1]))
        ask_price_pre = np.concatenate(([ask_price_post[0]], ask_price_post[:-1]))
        mids = 0.5 * (bid_price_pre + ask_price_pre)
        prices = self.messages["price"].to_numpy(dtype=float) if "price" in self.messages else mids
        times = self.messages["time"].to_numpy(dtype=float) if "time" in self.messages else np.arange(len(mids), dtype=float)
        strategy_times = _session_times(times, session_duration_seconds)
        decision_indices = _decision_indices(times, latency_ms)
        event_type = self.messages["event_type"].to_numpy(dtype=int) if "event_type" in self.messages else np.full(len(mids), 4)
        direction = self.messages["direction"].to_numpy(dtype=int) if "direction" in self.messages else np.zeros(len(mids), dtype=int)

        n_rows = min(len(mids), len(prices), len(times), len(event_type), len(direction))
        inventory = 0
        cash = 0.0
        inventory_path = np.zeros(n_rows + 1, dtype=int)
        fill_times: list[float] = []
        spreads: list[float] = []

        for idx in range(n_rows):
            t = float(strategy_times[idx])
            trade_price = float(prices[idx])
            decision_idx = int(decision_indices[idx])
            bid: float | None = None
            ask: float | None = None
            if decision_idx >= 0:
                decision_mid = float(mids[decision_idx])
                decision_t = float(strategy_times[decision_idx])
                bid_quote, ask_quote = strategy_fn(decision_mid, int(inventory), decision_t)
                bid, ask = _apply_post_only(
                    float(bid_quote),
                    float(ask_quote),
                    best_bid=float(bid_price_pre[decision_idx]),
                    best_ask=float(ask_price_pre[decision_idx]),
                    mode=post_only_mode,
                )

            if int(event_type[idx]) not in {4, 5}:
                inventory_path[idx + 1] = inventory
                continue

            if int(direction[idx]) > 0 and bid is not None and bid >= trade_price:
                cash -= bid
                inventory += 1
                fill_times.append(t)
                if ask is not None:
                    spreads.append(ask - bid)
            elif int(direction[idx]) < 0 and ask is not None and ask <= trade_price:
                cash += ask
                inventory -= 1
                fill_times.append(t)
                if bid is not None:
                    spreads.append(ask - bid)

            inventory_path[idx + 1] = inventory

        pnl = cash + inventory * float(mids[n_rows - 1])
        spread_realized = float(np.mean(spreads)) if spreads else 0.0
        return BacktestResult(
            pnl=float(pnl),
            inventory_path=inventory_path,
            fill_times=fill_times,
            spread_realized=spread_realized,
        )

    def _run_queue_aware_strategy(
        self,
        strategy_fn: Callable[[float, int, float], tuple[float, float]],
        session_duration_seconds: float,
        post_only_mode: Literal["reprice", "reject"],
        order_size: int,
        cancellation_rule: Literal["cancel-from-back", "proportional"],
        latency_ms: float,
    ) -> BacktestResult:
        if self.orderbook is None or self.messages is None:
            raise RuntimeError("LOBSTER data not loaded. Call load(...) first.")

        times = self.messages["time"].to_numpy(dtype=float) if "time" in self.messages else np.arange(len(self.orderbook), dtype=float)
        strategy_times = _session_times(times, session_duration_seconds)
        decision_indices = _decision_indices(times, latency_ms)
        prices = self.messages["price"].to_numpy(dtype=float) if "price" in self.messages else self.mid_price_series().to_numpy(dtype=float)
        sizes = self.messages["size"].to_numpy(dtype=float) if "size" in self.messages else np.ones(len(self.orderbook), dtype=float)
        event_type = self.messages["event_type"].to_numpy(dtype=int) if "event_type" in self.messages else np.full(len(self.orderbook), 4)
        direction = self.messages["direction"].to_numpy(dtype=int) if "direction" in self.messages else np.zeros(len(self.orderbook), dtype=int)

        if order_size < 1:
            raise ValueError("order_size must be positive")
        if cancellation_rule not in {"cancel-from-back", "proportional"}:
            raise ValueError(f"Unsupported cancellation rule: {cancellation_rule}")

        bid_price_post, bid_size_post = _side_levels(self.orderbook, "bid")
        ask_price_post, ask_size_post = _side_levels(self.orderbook, "ask")

        n_rows = min(
            len(times),
            len(prices),
            len(sizes),
            len(event_type),
            len(direction),
            bid_price_post.shape[0],
            ask_price_post.shape[0],
            bid_size_post.shape[0],
            ask_size_post.shape[0],
        )
        if n_rows == 0:
            return BacktestResult(
                pnl=0.0,
                inventory_path=np.zeros(1, dtype=int),
                fill_times=[],
                spread_realized=0.0,
            )

        bid_price_pre = _pre_event_levels(bid_price_post[:n_rows])
        ask_price_pre = _pre_event_levels(ask_price_post[:n_rows])
        bid_size_pre = _pre_event_levels(bid_size_post[:n_rows])
        ask_size_pre = _pre_event_levels(ask_size_post[:n_rows])
        mids = 0.5 * (bid_price_pre[:, 0] + ask_price_pre[:, 0])

        inventory = 0
        cash = 0.0
        inventory_path = np.zeros(n_rows + 1, dtype=int)
        fill_times: list[float] = []
        spreads: list[float] = []
        positions_at_fill: list[int] = []
        time_to_fill: list[float] = []
        cancel_volume = 0.0
        bid_order = _RestingOrder()
        ask_order = _RestingOrder()

        for idx in range(n_rows):
            t = float(strategy_times[idx])
            trade_price = float(prices[idx])
            event_size = max(int(round(float(sizes[idx]))), 0)
            decision_idx = int(decision_indices[idx])
            bid: float | None = None
            ask: float | None = None
            if decision_idx >= 0:
                decision_mid = float(mids[decision_idx])
                decision_t = float(strategy_times[decision_idx])
                bid_quote, ask_quote = strategy_fn(decision_mid, int(inventory), decision_t)
                bid, ask = _apply_post_only(
                    float(bid_quote),
                    float(ask_quote),
                    best_bid=float(bid_price_pre[decision_idx, 0]),
                    best_ask=float(ask_price_pre[decision_idx, 0]),
                    mode=post_only_mode,
                )
            bid_order = _refresh_order_state(
                order=bid_order,
                quote=bid,
                best_price=float(bid_price_pre[decision_idx, 0]) if decision_idx >= 0 else 0.0,
                level_prices=bid_price_pre[decision_idx] if decision_idx >= 0 else np.asarray([]),
                level_sizes=bid_size_pre[decision_idx] if decision_idx >= 0 else np.asarray([]),
                side="bid",
                placed_at=float(strategy_times[decision_idx]) if decision_idx >= 0 else t,
                order_size=order_size,
            )
            ask_order = _refresh_order_state(
                order=ask_order,
                quote=ask,
                best_price=float(ask_price_pre[decision_idx, 0]) if decision_idx >= 0 else 0.0,
                level_prices=ask_price_pre[decision_idx] if decision_idx >= 0 else np.asarray([]),
                level_sizes=ask_size_pre[decision_idx] if decision_idx >= 0 else np.asarray([]),
                side="ask",
                placed_at=float(strategy_times[decision_idx]) if decision_idx >= 0 else t,
                order_size=order_size,
            )

            current_event = int(event_type[idx])
            current_direction = int(direction[idx])
            if current_event in {4, 5} and event_size > 0:
                if current_direction > 0 and _event_matches_order(bid_order, trade_price):
                    position_before = int(bid_order.position or 0)
                    fill_size = _execute_against_order(bid_order, event_size)
                    if fill_size > 0:
                        cash -= float(bid_order.price) * fill_size
                        inventory += fill_size
                        fill_times.append(t)
                        if ask is not None and bid is not None:
                            spreads.append(ask - bid)
                        positions_at_fill.append(position_before)
                        if bid_order.placed_at is not None:
                            time_to_fill.append(max(t - bid_order.placed_at, 0.0))
                    if bid_order.remaining_size == 0:
                        bid_order = _RestingOrder()
                elif current_direction < 0 and _event_matches_order(ask_order, trade_price):
                    position_before = int(ask_order.position or 0)
                    fill_size = _execute_against_order(ask_order, event_size)
                    if fill_size > 0:
                        cash += float(ask_order.price) * fill_size
                        inventory -= fill_size
                        fill_times.append(t)
                        if ask is not None and bid is not None:
                            spreads.append(ask - bid)
                        positions_at_fill.append(position_before)
                        if ask_order.placed_at is not None:
                            time_to_fill.append(max(t - ask_order.placed_at, 0.0))
                    if ask_order.remaining_size == 0:
                        ask_order = _RestingOrder()
            elif current_event in {2, 3} and event_size > 0:
                if current_direction > 0:
                    cancel_volume += event_size
                    if _event_matches_order(bid_order, trade_price):
                        _cancel_ahead(
                            bid_order,
                            event_size,
                            level_size=_displayed_size_at_price(
                                bid_price_pre[idx],
                                bid_size_pre[idx],
                                trade_price,
                            ),
                            rule=cancellation_rule,
                        )
                elif current_direction < 0:
                    cancel_volume += event_size
                    if _event_matches_order(ask_order, trade_price):
                        _cancel_ahead(
                            ask_order,
                            event_size,
                            level_size=_displayed_size_at_price(
                                ask_price_pre[idx],
                                ask_size_pre[idx],
                                trade_price,
                            ),
                            rule=cancellation_rule,
                        )

            inventory_path[idx + 1] = inventory

        pnl = cash + inventory * float(mids[n_rows - 1])
        spread_realized = float(np.mean(spreads)) if spreads else 0.0
        horizon = _event_horizon(times[:n_rows])
        latencies = np.asarray(time_to_fill, dtype=float)
        return BacktestResult(
            pnl=float(pnl),
            inventory_path=inventory_path,
            fill_times=fill_times,
            spread_realized=spread_realized,
            avg_position_at_fill=float(np.mean(positions_at_fill)) if positions_at_fill else 0.0,
            cancellation_rate=float(cancel_volume / max(horizon, 1e-12)),
            time_to_fill_mean=float(np.mean(latencies)) if latencies.size else 0.0,
            time_to_fill_p50=float(np.percentile(latencies, 50)) if latencies.size else 0.0,
            time_to_fill_p90=float(np.percentile(latencies, 90)) if latencies.size else 0.0,
        )


def _refresh_order_state(
    *,
    order: _RestingOrder,
    quote: float | None,
    best_price: float,
    level_prices: np.ndarray,
    level_sizes: np.ndarray,
    side: str,
    placed_at: float,
    order_size: int,
) -> _RestingOrder:
    if quote is None:
        return _RestingOrder()
    quote_level = round(float(quote), 2)
    best_level = round(float(best_price), 2)

    if order.price is not None and np.isclose(order.price, quote_level) and order.position is not None:
        return order
    improves_touch = quote_level > best_level if side == "bid" else quote_level < best_level
    initial_position = 0 if improves_touch else _displayed_size_at_price(level_prices, level_sizes, quote_level)
    if initial_position is None:
        return _RestingOrder()
    return _RestingOrder(
        price=quote_level,
        position=initial_position,
        remaining_size=order_size,
        placed_at=placed_at,
    )


def _event_matches_order(order: _RestingOrder, event_price: float) -> bool:
    return order.price is not None and order.position is not None and np.isclose(order.price, event_price)


def _execute_against_order(order: _RestingOrder, executed_size: int) -> int:
    if order.position is None or order.remaining_size <= 0:
        return 0
    consumed_ahead = min(max(executed_size, 0), order.position)
    order.position -= consumed_ahead
    available_for_order = max(executed_size - consumed_ahead, 0)
    fill_size = min(available_for_order, order.remaining_size)
    order.remaining_size -= fill_size
    return fill_size


def _cancel_ahead(
    order: _RestingOrder,
    cancelled_size: int,
    *,
    level_size: int | None,
    rule: Literal["cancel-from-back", "proportional"],
) -> None:
    if order.position is None or order.position <= 0:
        return
    if rule == "cancel-from-back":
        cancelled_ahead = cancelled_size
    else:
        displayed = max(int(level_size or 0), 1)
        ahead_fraction = min(order.position / displayed, 1.0)
        cancelled_ahead = int(round(cancelled_size * ahead_fraction))
    order.position = max(order.position - min(cancelled_ahead, order.position), 0)


def _displayed_size_at_price(
    level_prices: np.ndarray,
    level_sizes: np.ndarray,
    price: float,
) -> int | None:
    matches = np.flatnonzero(np.isclose(np.asarray(level_prices, dtype=float), float(price)))
    if matches.size == 0:
        return None
    return int(max(float(np.asarray(level_sizes, dtype=float)[int(matches[0])]), 0.0))


def _side_levels(frame: pd.DataFrame, side: Literal["bid", "ask"]) -> tuple[np.ndarray, np.ndarray]:
    price_columns = [column for column in frame.columns if column.startswith(f"{side}_price_")]
    size_columns = [column for column in frame.columns if column.startswith(f"{side}_size_")]
    if not price_columns or len(price_columns) != len(size_columns):
        raise ValueError(f"LOBSTER order book has no complete {side} levels")
    return (
        frame[price_columns].to_numpy(dtype=float),
        frame[size_columns].to_numpy(dtype=int),
    )


def _pre_event_levels(post_event_levels: np.ndarray) -> np.ndarray:
    values = np.asarray(post_event_levels)
    if values.shape[0] == 0:
        return values.copy()
    return np.concatenate((values[:1], values[:-1]), axis=0)


def _event_horizon(times: np.ndarray) -> float:
    if times.size < 2:
        return float(max(times.size, 1))
    horizon = float(times[-1] - times[0])
    return horizon if horizon > 0 else float(times.size)


def _session_times(times: np.ndarray, session_duration_seconds: float) -> np.ndarray:
    if session_duration_seconds <= 0.0:
        raise ValueError("session_duration_seconds must be positive")
    values = np.asarray(times, dtype=float)
    if values.size == 0:
        return values.copy()
    return np.clip(values - values[0], 0.0, float(session_duration_seconds))


def _apply_post_only(
    bid: float,
    ask: float,
    *,
    best_bid: float,
    best_ask: float,
    mode: Literal["reprice", "reject"],
) -> tuple[float | None, float | None]:
    if mode not in {"reprice", "reject"}:
        raise ValueError(f"Unsupported post-only mode: {mode}")
    if bid >= best_ask:
        bid = best_bid if mode == "reprice" else None
    if ask <= best_bid:
        ask = best_ask if mode == "reprice" else None
    return bid, ask


def _decision_indices(times: np.ndarray, latency_ms: float) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=int)
    activation_cutoff = values - float(latency_ms) / 1_000.0
    return np.searchsorted(values, activation_cutoff + 1e-12, side="right") - 1
