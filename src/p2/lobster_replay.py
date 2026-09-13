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
        names = ["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"]
        if frame.shape[1] > 4:
            names.extend([f"col_{idx}" for idx in range(4, frame.shape[1])])
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
    ) -> BacktestResult:
        if use_queue_position:
            return self._run_queue_aware_strategy(strategy_fn, session_duration_seconds, post_only_mode)
        return self._run_instantaneous_strategy(strategy_fn, session_duration_seconds, post_only_mode)

    def _run_instantaneous_strategy(
        self,
        strategy_fn: Callable[[float, int, float], tuple[float, float]],
        session_duration_seconds: float,
        post_only_mode: Literal["reprice", "reject"],
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
        event_type = self.messages["event_type"].to_numpy(dtype=int) if "event_type" in self.messages else np.full(len(mids), 4)
        direction = self.messages["direction"].to_numpy(dtype=int) if "direction" in self.messages else np.zeros(len(mids), dtype=int)

        n_rows = min(len(mids), len(prices), len(times), len(event_type), len(direction))
        inventory = 0
        cash = 0.0
        inventory_path = np.zeros(n_rows + 1, dtype=int)
        fill_times: list[float] = []
        spreads: list[float] = []

        for idx in range(n_rows):
            mid = float(mids[idx])
            t = float(strategy_times[idx])
            trade_price = float(prices[idx])
            bid, ask = strategy_fn(mid, int(inventory), t)
            bid, ask = _apply_post_only(
                float(bid),
                float(ask),
                best_bid=float(bid_price_pre[idx]),
                best_ask=float(ask_price_pre[idx]),
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
    ) -> BacktestResult:
        if self.orderbook is None or self.messages is None:
            raise RuntimeError("LOBSTER data not loaded. Call load(...) first.")

        times = self.messages["time"].to_numpy(dtype=float) if "time" in self.messages else np.arange(len(self.orderbook), dtype=float)
        strategy_times = _session_times(times, session_duration_seconds)
        prices = self.messages["price"].to_numpy(dtype=float) if "price" in self.messages else self.mid_price_series().to_numpy(dtype=float)
        sizes = self.messages["size"].to_numpy(dtype=float) if "size" in self.messages else np.ones(len(self.orderbook), dtype=float)
        event_type = self.messages["event_type"].to_numpy(dtype=int) if "event_type" in self.messages else np.full(len(self.orderbook), 4)
        direction = self.messages["direction"].to_numpy(dtype=int) if "direction" in self.messages else np.zeros(len(self.orderbook), dtype=int)

        bid_price_post = self.orderbook["bid_price_1"].to_numpy(dtype=float)
        ask_price_post = self.orderbook["ask_price_1"].to_numpy(dtype=float)
        bid_size_post = self.orderbook["bid_size_1"].to_numpy(dtype=int)
        ask_size_post = self.orderbook["ask_size_1"].to_numpy(dtype=int)

        n_rows = min(
            len(times),
            len(prices),
            len(sizes),
            len(event_type),
            len(direction),
            len(bid_price_post),
            len(ask_price_post),
            len(bid_size_post),
            len(ask_size_post),
        )
        if n_rows == 0:
            return BacktestResult(
                pnl=0.0,
                inventory_path=np.zeros(1, dtype=int),
                fill_times=[],
                spread_realized=0.0,
            )

        bid_price_pre = np.concatenate(([bid_price_post[0]], bid_price_post[: n_rows - 1]))
        ask_price_pre = np.concatenate(([ask_price_post[0]], ask_price_post[: n_rows - 1]))
        bid_size_pre = np.concatenate(([bid_size_post[0]], bid_size_post[: n_rows - 1]))
        ask_size_pre = np.concatenate(([ask_size_post[0]], ask_size_post[: n_rows - 1]))
        mids = 0.5 * (bid_price_pre + ask_price_pre)

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
            mid = float(mids[idx])
            t = float(strategy_times[idx])
            trade_price = float(prices[idx])
            event_size = max(int(round(float(sizes[idx]))), 0)
            bid, ask = strategy_fn(mid, int(inventory), t)
            best_bid = float(bid_price_pre[idx])
            best_ask = float(ask_price_pre[idx])
            bid, ask = _apply_post_only(
                float(bid),
                float(ask),
                best_bid=best_bid,
                best_ask=best_ask,
                mode=post_only_mode,
            )
            best_bid_size = int(max(bid_size_pre[idx], 0))
            best_ask_size = int(max(ask_size_pre[idx], 0))

            bid_order = _refresh_order_state(
                order=bid_order,
                quote=bid,
                best_price=best_bid,
                best_size=best_bid_size,
                side="bid",
                placed_at=t,
            )
            ask_order = _refresh_order_state(
                order=ask_order,
                quote=ask,
                best_price=best_ask,
                best_size=best_ask_size,
                side="ask",
                placed_at=t,
            )

            current_event = int(event_type[idx])
            current_direction = int(direction[idx])
            if current_event in {4, 5} and event_size > 0:
                if current_direction > 0 and bid_order.position is not None and bid_order.price is not None and bid_order.price >= trade_price:
                    if event_size > bid_order.position:
                        cash -= float(bid_order.price)
                        inventory += 1
                        fill_times.append(t)
                        if ask is not None and bid is not None:
                            spreads.append(ask - bid)
                        positions_at_fill.append(bid_order.position)
                        if bid_order.placed_at is not None:
                            time_to_fill.append(max(t - bid_order.placed_at, 0.0))
                        bid_order = _RestingOrder()
                    else:
                        bid_order.position = max(bid_order.position - event_size, 0)
                elif current_direction < 0 and ask_order.position is not None and ask_order.price is not None and ask_order.price <= trade_price:
                    if event_size > ask_order.position:
                        cash += float(ask_order.price)
                        inventory -= 1
                        fill_times.append(t)
                        if ask is not None and bid is not None:
                            spreads.append(ask - bid)
                        positions_at_fill.append(ask_order.position)
                        if ask_order.placed_at is not None:
                            time_to_fill.append(max(t - ask_order.placed_at, 0.0))
                        ask_order = _RestingOrder()
                    else:
                        ask_order.position = max(ask_order.position - event_size, 0)
            elif current_event in {2, 3} and event_size > 0:
                if current_direction > 0 and np.isclose(trade_price, best_bid):
                    cancel_volume += event_size
                    if bid_order.position is not None and bid_order.price is not None and np.isclose(bid_order.price, best_bid):
                        bid_order.position = max(bid_order.position - event_size, 0)
                elif current_direction < 0 and np.isclose(trade_price, best_ask):
                    cancel_volume += event_size
                    if ask_order.position is not None and ask_order.price is not None and np.isclose(ask_order.price, best_ask):
                        ask_order.position = max(ask_order.position - event_size, 0)

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
    best_size: int,
    side: str,
    placed_at: float,
) -> _RestingOrder:
    if quote is None:
        return _RestingOrder()
    quote_level = round(float(quote), 2)
    best_level = round(float(best_price), 2)

    if side == "bid":
        active = quote_level >= best_level
        initial_position = 0 if quote_level > best_level else max(best_size, 0)
    else:
        active = quote_level <= best_level
        initial_position = 0 if quote_level < best_level else max(best_size, 0)

    if not active:
        return _RestingOrder()
    if order.price is not None and np.isclose(order.price, quote_level) and order.position is not None:
        return order
    return _RestingOrder(price=quote_level, position=initial_position, placed_at=placed_at)


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
