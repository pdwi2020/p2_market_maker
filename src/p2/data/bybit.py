"""Bybit level-two order-book reconstruction."""

from __future__ import annotations

import argparse
import bisect
import json
import time
from dataclasses import asdict, dataclass
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from p2.config import load_config


BOOK_EVENT = 0
TRADE_EVENT = 1
DEFAULT_TOP_N = 20


@dataclass(frozen=True)
class BybitQualityReport:
    """Numeric reconstruction diagnostics for one symbol-day."""

    symbol: str
    date: str
    book_events: int
    trade_events: int
    gaps: int
    invalid_book_seconds: float
    crossed_count: int
    trades_outside_book: int
    trades_checked: int
    runtime_seconds: float


@dataclass(frozen=True)
class BybitDayResult:
    """Paths and diagnostics produced by one reconstruction."""

    cache_path: Path
    quality_path: Path
    quality: BybitQualityReport


class _BookSide:
    """Price-indexed depth with an ascending price index."""

    __slots__ = ("prices", "sizes")

    def __init__(self) -> None:
        self.prices: list[float] = []
        self.sizes: dict[float, float] = {}

    def reset(self, levels: Iterable[Sequence[str]]) -> None:
        sizes: dict[float, float] = {}
        for raw_price, raw_size in levels:
            price = float(raw_price)
            size = float(raw_size)
            if size > 0.0:
                sizes[price] = size
        self.sizes = sizes
        self.prices = sorted(sizes)

    def update(self, levels: Iterable[Sequence[str]]) -> None:
        for raw_price, raw_size in levels:
            price = float(raw_price)
            size = float(raw_size)
            current = self.sizes.get(price)
            if size == 0.0:
                if current is not None:
                    del self.sizes[price]
                    index = bisect.bisect_left(self.prices, price)
                    if index == len(self.prices) or self.prices[index] != price:
                        raise RuntimeError("price index is inconsistent with depth")
                    self.prices.pop(index)
            elif size > 0.0:
                self.sizes[price] = size
                if current is None:
                    bisect.insort(self.prices, price)
            else:
                raise ValueError("book size cannot be negative")

    def best(self, *, reverse: bool) -> tuple[float, float]:
        if not self.prices:
            return np.nan, np.nan
        price = self.prices[-1] if reverse else self.prices[0]
        return price, self.sizes[price]

    def write_top(
        self,
        price_output: np.ndarray,
        size_output: np.ndarray,
        row: int,
        *,
        reverse: bool,
    ) -> None:
        prices = self.prices
        if reverse:
            selected = reversed(prices[max(0, len(prices) - price_output.shape[1]) :])
        else:
            selected = iter(prices[: price_output.shape[1]])
        for column, price in enumerate(selected):
            price_output[row, column] = price
            size_output[row, column] = self.sizes[price]


def _day_bounds_ms(day: str) -> tuple[int, int]:
    parsed = date_type.fromisoformat(day)
    start = datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1_000)
    return start_ms, start_ms + 86_400_000


def _list_array(values: np.ndarray) -> pa.ListArray:
    width = values.shape[1]
    offsets = np.arange(len(values) + 1, dtype=np.int64) * width
    return pa.ListArray.from_arrays(offsets, pa.array(values.reshape(-1)))


def _book_event_table(
    timestamps: np.ndarray,
    receive_times: np.ndarray,
    update_ids: np.ndarray,
    valid: np.ndarray,
    best_bid: np.ndarray,
    best_bid_size: np.ndarray,
    best_ask: np.ndarray,
    best_ask_size: np.ndarray,
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
) -> pa.Table:
    count = len(timestamps)
    event_type = pa.DictionaryArray.from_arrays(
        pa.array(np.zeros(count, dtype=np.int8)), pa.array(["book", "trade"])
    )
    return pa.table(
        {
            "time": pa.array(timestamps, type=pa.timestamp("ms", tz="UTC")),
            "receive_time": pa.array(receive_times, type=pa.timestamp("ms", tz="UTC")),
            "event_type": event_type,
            "update_id": update_ids,
            "book_valid": valid,
            "best_bid": best_bid,
            "best_bid_size": best_bid_size,
            "best_ask": best_ask,
            "best_ask_size": best_ask_size,
            "bid_prices": _list_array(bid_prices),
            "bid_sizes": _list_array(bid_sizes),
            "ask_prices": _list_array(ask_prices),
            "ask_sizes": _list_array(ask_sizes),
            "trade_price": pa.nulls(count, pa.float64()),
            "trade_volume": pa.nulls(count, pa.float64()),
            "trade_side": pa.nulls(count, pa.string()),
        }
    )


def _trade_event_table(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: Sequence[str],
    top_n: int,
) -> pa.Table:
    count = len(timestamps)
    event_type = pa.DictionaryArray.from_arrays(
        pa.array(np.ones(count, dtype=np.int8)), pa.array(["book", "trade"])
    )
    empty_levels = pa.nulls(count, pa.list_(pa.float64()))
    return pa.table(
        {
            "time": pa.array(timestamps, type=pa.timestamp("ms", tz="UTC")),
            "receive_time": pa.nulls(count, pa.timestamp("ms", tz="UTC")),
            "event_type": event_type,
            "update_id": pa.nulls(count, pa.int64()),
            "book_valid": pa.nulls(count, pa.bool_()),
            "best_bid": pa.nulls(count, pa.float64()),
            "best_bid_size": pa.nulls(count, pa.float64()),
            "best_ask": pa.nulls(count, pa.float64()),
            "best_ask_size": pa.nulls(count, pa.float64()),
            "bid_prices": empty_levels,
            "bid_sizes": empty_levels,
            "ask_prices": empty_levels,
            "ask_sizes": empty_levels,
            "trade_price": prices,
            "trade_volume": volumes,
            "trade_side": pa.array(sides, pa.string()),
        }
    )


def _read_trades(path: Path, start_ms: int, end_ms: int) -> tuple[np.ndarray, ...]:
    table = pq.read_table(path, columns=["timestamp", "price", "volume", "side"])
    timestamps = table["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
    keep = (timestamps >= start_ms) & (timestamps < end_ms)
    timestamps = timestamps[keep]
    prices = table["price"].to_numpy(zero_copy_only=False)[keep].astype(np.float64)
    volumes = table["volume"].to_numpy(zero_copy_only=False)[keep].astype(np.float64)
    sides = np.asarray(table["side"].to_pylist(), dtype=object)[keep]
    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], prices[order], volumes[order], sides[order]


def _outside_book_count(
    trade_times: np.ndarray,
    trade_prices: np.ndarray,
    book_times: np.ndarray,
    valid: np.ndarray,
    best_bid: np.ndarray,
    best_ask: np.ndarray,
) -> tuple[int, int]:
    if len(book_times) == 0:
        return 0, 0
    positions = np.searchsorted(book_times, trade_times, side="right") - 1
    has_book = positions >= 0
    safe_positions = np.maximum(positions, 0)
    checked = has_book & valid[safe_positions]
    checked &= np.isfinite(best_bid[safe_positions]) & np.isfinite(best_ask[safe_positions])
    outside = checked & (
        (trade_prices < best_bid[safe_positions]) | (trade_prices > best_ask[safe_positions])
    )
    return int(np.count_nonzero(outside)), int(np.count_nonzero(checked))


def reconstruct_bybit_day(
    orderbook_path: str | Path,
    trades_path: str | Path,
    cache_dir: str | Path,
    symbol: str,
    day: str,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> BybitDayResult:
    """Reconstruct, validate, merge, and cache one UTC symbol-day."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    started = time.perf_counter()
    orderbook_path = Path(orderbook_path)
    trades_path = Path(trades_path)
    start_ms, end_ms = _day_bounds_ms(day)

    parquet_file = pq.ParquetFile(orderbook_path)
    capacity = parquet_file.metadata.num_rows
    timestamps = np.empty(capacity, dtype=np.int64)
    receive_times = np.empty(capacity, dtype=np.int64)
    update_ids = np.empty(capacity, dtype=np.int64)
    valid = np.zeros(capacity, dtype=np.bool_)
    best_bid = np.full(capacity, np.nan, dtype=np.float64)
    best_bid_size = np.full(capacity, np.nan, dtype=np.float64)
    best_ask = np.full(capacity, np.nan, dtype=np.float64)
    best_ask_size = np.full(capacity, np.nan, dtype=np.float64)
    bid_prices = np.full((capacity, top_n), np.nan, dtype=np.float64)
    bid_sizes = np.full((capacity, top_n), np.nan, dtype=np.float64)
    ask_prices = np.full((capacity, top_n), np.nan, dtype=np.float64)
    ask_sizes = np.full((capacity, top_n), np.nan, dtype=np.float64)

    bids = _BookSide()
    asks = _BookSide()
    output_row = 0
    last_update_id: int | None = None
    invalid_since = start_ms
    invalid_ms = 0
    gaps = 0
    crossed_count = 0

    columns = ["ts", "cts", "type", "update_id", "bids", "asks"]
    for batch in parquet_file.iter_batches(columns=columns, batch_size=65_536):
        data = batch.to_pydict()
        for receive_time, timestamp, event_type, update_id, raw_bids, raw_asks in zip(
            data["ts"],
            data["cts"],
            data["type"],
            data["update_id"],
            data["bids"],
            data["asks"],
            strict=True,
        ):
            timestamp = int(timestamp)
            receive_time = int(receive_time)
            if timestamp < start_ms or timestamp >= end_ms:
                continue
            if output_row and timestamp < timestamps[output_row - 1]:
                raise ValueError("order-book event times must be nondecreasing")
            update_id = int(update_id)
            is_snapshot = event_type == "snapshot"
            if is_snapshot:
                bids.reset(json.loads(raw_bids))
                asks.reset(json.loads(raw_asks))
                if invalid_since is not None:
                    invalid_ms += max(0, timestamp - invalid_since)
                    invalid_since = None
                is_valid = True
            else:
                is_valid = invalid_since is None
                if last_update_id is not None and update_id != last_update_id + 1:
                    gaps += 1
                    if invalid_since is None:
                        invalid_since = timestamp
                    is_valid = False
                if is_valid:
                    bids.update(json.loads(raw_bids))
                    asks.update(json.loads(raw_asks))
            last_update_id = update_id

            timestamps[output_row] = timestamp
            receive_times[output_row] = receive_time
            update_ids[output_row] = update_id
            valid[output_row] = is_valid
            if is_valid:
                row_bid, row_bid_size = bids.best(reverse=True)
                row_ask, row_ask_size = asks.best(reverse=False)
                if row_bid >= row_ask:
                    crossed_count += 1
                    raise ValueError(
                        f"crossed book for {symbol} at {timestamp}: {row_bid} >= {row_ask}"
                    )
                best_bid[output_row] = row_bid
                best_bid_size[output_row] = row_bid_size
                best_ask[output_row] = row_ask
                best_ask_size[output_row] = row_ask_size
                bids.write_top(bid_prices, bid_sizes, output_row, reverse=True)
                asks.write_top(ask_prices, ask_sizes, output_row, reverse=False)
            output_row += 1

    if invalid_since is not None:
        invalid_ms += max(0, end_ms - invalid_since)

    timestamps = timestamps[:output_row]
    receive_times = receive_times[:output_row]
    update_ids = update_ids[:output_row]
    valid = valid[:output_row]
    best_bid = best_bid[:output_row]
    best_bid_size = best_bid_size[:output_row]
    best_ask = best_ask[:output_row]
    best_ask_size = best_ask_size[:output_row]
    bid_prices = bid_prices[:output_row]
    bid_sizes = bid_sizes[:output_row]
    ask_prices = ask_prices[:output_row]
    ask_sizes = ask_sizes[:output_row]

    trade_times, trade_prices, trade_volumes, trade_sides = _read_trades(
        trades_path, start_ms, end_ms
    )
    outside_count, checked_count = _outside_book_count(
        trade_times, trade_prices, timestamps, valid, best_bid, best_ask
    )
    book_table = _book_event_table(
        timestamps,
        receive_times,
        update_ids,
        valid,
        best_bid,
        best_bid_size,
        best_ask,
        best_ask_size,
        bid_prices,
        bid_sizes,
        ask_prices,
        ask_sizes,
    )
    trade_table = _trade_event_table(
        trade_times, trade_prices, trade_volumes, trade_sides, top_n
    )
    merged = pa.concat_tables([book_table, trade_table])
    merged = merged.take(pc.sort_indices(merged, sort_keys=[("time", "ascending")]))

    destination = Path(cache_dir) / "bybit" / symbol
    destination.mkdir(parents=True, exist_ok=True)
    cache_path = destination / f"{day}.parquet"
    quality_path = destination / f"{day}.quality.json"
    pq.write_table(merged, cache_path, compression="zstd")
    report = BybitQualityReport(
        symbol=symbol,
        date=day,
        book_events=output_row,
        trade_events=len(trade_times),
        gaps=gaps,
        invalid_book_seconds=invalid_ms / 1_000.0,
        crossed_count=crossed_count,
        trades_outside_book=outside_count,
        trades_checked=checked_count,
        runtime_seconds=time.perf_counter() - started,
    )
    quality_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    return BybitDayResult(cache_path=cache_path, quality_path=quality_path, quality=report)


def find_bybit_files(data_root: str | Path, symbol: str, day: str) -> tuple[Path, Path]:
    """Locate one day in either the sample tree or full-history tree."""
    root = Path(data_root)
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
    for book_root, trade_root in layouts:
        orderbook_path = book_root / f"date={day}" / "orderbook.parquet"
        trades_path = trade_root / f"date={day}" / "trades.parquet"
        if orderbook_path.is_file() and trades_path.is_file():
            return orderbook_path, trades_path
    raise FileNotFoundError(f"Bybit files not found for {symbol} on {day} under {root}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruct Bybit level-two data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", action="append", required=True, dest="dates")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    config = load_config()
    source_root = args.source_root or config.paths.data_dir
    cache_dir = args.cache_dir or config.paths.cache_dir
    for day in args.dates:
        orderbook_path, trades_path = find_bybit_files(source_root, args.symbol, day)
        result = reconstruct_bybit_day(
            orderbook_path, trades_path, cache_dir, args.symbol, day
        )
        print(json.dumps(asdict(result.quality), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
