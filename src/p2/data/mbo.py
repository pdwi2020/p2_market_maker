"""Order-level reconstruction and FIFO queue simulation for MBO data."""

from __future__ import annotations

import argparse
import heapq
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from p2.config import load_config


DAY_NS = 86_400_000_000_000
RTH_START_NS = 52_200_000_000_000
RTH_END_NS = 75_600_000_000_000
LAST_EVENT_FLAG = 128

_DUPLICATE_ADD = 1
_MISSING_ORDER = 2
_OVER_CANCEL = 4
_REFERENCE_MISMATCH = 8


@dataclass(frozen=True, slots=True)
class MBORecord:
    """Fields required to apply one order-level event."""

    ts_event_ns: int
    action: str
    side: str
    price: float
    size: int
    order_id: int
    flags: int = LAST_EVENT_FLAG
    ts_recv_ns: int | None = None

    @property
    def index_ns(self) -> int:
        """Return the primary timestamp used to order this record."""
        return self.ts_event_ns if self.ts_recv_ns is None else self.ts_recv_ns


@dataclass(slots=True)
class RestingOrder:
    """Current state of one exchange order."""

    side: str
    price: float
    size: int


@dataclass(frozen=True, slots=True)
class RestingFill:
    """A counterfactual fill of a submitted resting order."""

    time: pd.Timestamp
    price: float
    size: int


@dataclass(frozen=True)
class MBOQualityReport:
    """Numeric reconstruction diagnostics for one UTC date."""

    symbol: str
    date: str
    rows: int
    rth_rows: int
    action_counts: dict[str, int]
    duplicate_adds: int
    missing_order_events: int
    over_cancels: int
    reference_mismatches: int
    crossed_count: int
    outside_rth_crossed_count: int
    resets: int


@dataclass(frozen=True)
class MBOReconstructionResult:
    """Outputs produced by a complete order-level replay."""

    quality_paths: tuple[Path, ...]
    reports: tuple[MBOQualityReport, ...]
    final_order_count: int
    runtime_seconds: float


@dataclass
class _QualityAccumulator:
    rows: int = 0
    rth_rows: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    duplicate_adds: int = 0
    missing_order_events: int = 0
    over_cancels: int = 0
    reference_mismatches: int = 0
    crossed_count: int = 0
    outside_rth_crossed_count: int = 0
    resets: int = 0


def normalize_side(side: str) -> str:
    """Normalize common bid/ask labels to the MBO side codes."""
    normalized = side.strip().lower()
    if normalized in {"b", "bid", "buy"}:
        return "B"
    if normalized in {"a", "ask", "sell"}:
        return "A"
    raise ValueError(f"unsupported resting side: {side}")


def is_rth(ts_event_ns: int) -> bool:
    """Return whether a timestamp is in January's 14:30-21:00 UTC RTH."""
    timestamp = datetime.fromtimestamp(ts_event_ns / 1_000_000_000, tz=timezone.utc)
    within_hours = RTH_START_NS <= ts_event_ns % DAY_NS < RTH_END_NS
    return timestamp.month == 1 and within_hours


class MBOBook:
    """Order-level book with FIFO queues at each price."""

    def __init__(self) -> None:
        self.orders: dict[int, RestingOrder] = {}
        self._levels: dict[str, dict[float, dict[int, None]]] = {"B": {}, "A": {}}
        self._price_heaps: dict[str, list[float]] = {"B": [], "A": []}

    def clear(self) -> None:
        self.orders.clear()
        self._levels["B"].clear()
        self._levels["A"].clear()
        self._price_heaps["B"].clear()
        self._price_heaps["A"].clear()

    def _queue(self, side: str, price: float, *, create: bool) -> dict[int, None] | None:
        levels = self._levels[side]
        queue = levels.get(price)
        if queue is None and create:
            queue = {}
            levels[price] = queue
            heapq.heappush(self._price_heaps[side], -price if side == "B" else price)
        return queue

    def _remove(self, order_id: int) -> RestingOrder:
        order = self.orders.pop(order_id)
        queue = self._levels[order.side][order.price]
        del queue[order_id]
        if not queue:
            del self._levels[order.side][order.price]
        return order

    def _insert(self, order_id: int, side: str, price: float, size: int) -> None:
        queue = self._queue(side, price, create=True)
        if queue is None:
            raise RuntimeError("price queue was not created")
        queue[order_id] = None
        self.orders[order_id] = RestingOrder(side=side, price=price, size=size)

    def _best(self, side: str) -> float | None:
        heap = self._price_heaps[side]
        levels = self._levels[side]
        while heap:
            price = -heap[0] if side == "B" else heap[0]
            if price in levels:
                return price
            heapq.heappop(heap)
        return None

    @property
    def best_bid(self) -> float | None:
        return self._best("B")

    @property
    def best_ask(self) -> float | None:
        return self._best("A")

    @property
    def crossed(self) -> bool:
        bid = self.best_bid
        ask = self.best_ask
        return bid is not None and ask is not None and bid >= ask

    def level_order_ids(self, side: str, price: float) -> tuple[int, ...]:
        queue = self._queue(normalize_side(side), float(price), create=False)
        return tuple(queue) if queue is not None else ()

    def apply(self, record: MBORecord) -> int:
        """Apply an event and return a bit mask of data-quality anomalies."""
        action = record.action
        if action == "R":
            self.clear()
            return 0
        if action in {"T", "N"}:
            return 0
        if action == "F":
            return 0 if record.order_id in self.orders else _MISSING_ORDER
        if action not in {"A", "C", "M"}:
            raise ValueError(f"unsupported MBO action: {action}")

        side = normalize_side(record.side)
        price = float(record.price)
        size = int(record.size)
        if size < 0:
            raise ValueError("order size cannot be negative")

        if action == "A":
            anomaly = 0
            if record.order_id in self.orders:
                self._remove(record.order_id)
                anomaly |= _DUPLICATE_ADD
            if size > 0:
                self._insert(record.order_id, side, price, size)
            return anomaly

        current = self.orders.get(record.order_id)
        if current is None:
            return _MISSING_ORDER
        anomaly = 0
        if action == "C":
            if current.side != side or current.price != price:
                anomaly |= _REFERENCE_MISMATCH
            if size > current.size:
                anomaly |= _OVER_CANCEL
            remaining = current.size - size
            if remaining <= 0:
                self._remove(record.order_id)
            else:
                current.size = remaining
            return anomaly

        loses_priority = side != current.side or price != current.price or size > current.size
        if size == 0:
            self._remove(record.order_id)
        elif loses_priority:
            self._remove(record.order_id)
            self._insert(record.order_id, side, price, size)
        else:
            current.size = size
        return anomaly


def iter_mbo_records(
    source: str | Path,
    *,
    symbol: str | None = "ES.c.0",
    rth_only: bool = False,
    batch_size: int = 262_144,
) -> Iterator[MBORecord]:
    """Stream normalized records from a Databento MBO parquet file."""
    columns = [
        "ts_event",
        "ts_recv",
        "action",
        "side",
        "price",
        "size",
        "order_id",
        "flags",
    ]
    if symbol is not None:
        columns.append("symbol")
    parquet_file = pq.ParquetFile(Path(source))
    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        timestamp_values = batch.column("ts_event").to_numpy(zero_copy_only=False)
        timestamps = timestamp_values.astype("datetime64[ns]").astype(np.int64)
        receive_values = batch.column("ts_recv").to_numpy(zero_copy_only=False)
        receive_times = receive_values.astype("datetime64[ns]").astype(np.int64)
        actions = batch.column("action").to_pylist()
        sides = batch.column("side").to_pylist()
        prices = batch.column("price").to_numpy(zero_copy_only=False)
        sizes = batch.column("size").to_numpy(zero_copy_only=False)
        order_ids = batch.column("order_id").to_numpy(zero_copy_only=False)
        flags = batch.column("flags").to_numpy(zero_copy_only=False)
        symbols = batch.column("symbol").to_pylist() if symbol is not None else None
        for index, timestamp in enumerate(timestamps):
            timestamp = int(timestamp)
            receive_time = int(receive_times[index])
            if symbols is not None and symbols[index] != symbol:
                continue
            if rth_only and not is_rth(receive_time):
                continue
            raw_price = prices[index]
            price = float(raw_price) if raw_price is not None else np.nan
            yield MBORecord(
                ts_event_ns=timestamp,
                action=actions[index],
                side=sides[index],
                price=price,
                size=int(sizes[index]),
                order_id=int(order_ids[index]),
                flags=int(flags[index]),
                ts_recv_ns=receive_time,
            )


def _date_from_day_number(day_number: int) -> date_type:
    seconds = day_number * 86_400
    return datetime.fromtimestamp(seconds, tz=timezone.utc).date()


def reconstruct_mbo(
    source: str | Path,
    cache_dir: str | Path,
    *,
    symbol: str = "ES.c.0",
) -> MBOReconstructionResult:
    """Replay an MBO file and write one numeric quality report per UTC date."""
    started = time.perf_counter()
    book = MBOBook()
    accumulators: dict[date_type, _QualityAccumulator] = {}
    last_timestamp = -1
    current_day_number = -1
    event_date: date_type | None = None
    accumulator: _QualityAccumulator | None = None
    for record in iter_mbo_records(source, symbol=symbol):
        if record.index_ns < last_timestamp:
            raise ValueError("MBO event times must be nondecreasing")
        last_timestamp = record.index_ns
        day_number = record.index_ns // DAY_NS
        if day_number != current_day_number:
            current_day_number = day_number
            event_date = _date_from_day_number(day_number)
            accumulator = accumulators.setdefault(event_date, _QualityAccumulator())
        if accumulator is None or event_date is None:
            raise RuntimeError("quality accumulator was not initialized")
        accumulator.rows += 1
        accumulator.action_counts[record.action] += 1
        in_rth = event_date.month == 1 and (
            RTH_START_NS <= record.index_ns % DAY_NS < RTH_END_NS
        )
        if in_rth:
            accumulator.rth_rows += 1
        anomaly = book.apply(record)
        accumulator.duplicate_adds += bool(anomaly & _DUPLICATE_ADD)
        accumulator.missing_order_events += bool(anomaly & _MISSING_ORDER)
        accumulator.over_cancels += bool(anomaly & _OVER_CANCEL)
        accumulator.reference_mismatches += bool(anomaly & _REFERENCE_MISMATCH)
        accumulator.resets += record.action == "R"
        if record.flags & LAST_EVENT_FLAG and book.crossed:
            if in_rth:
                accumulator.crossed_count += 1
            else:
                accumulator.outside_rth_crossed_count += 1

    destination = Path(cache_dir) / "mbo" / symbol
    destination.mkdir(parents=True, exist_ok=True)
    reports: list[MBOQualityReport] = []
    quality_paths: list[Path] = []
    for event_date, accumulator in sorted(accumulators.items()):
        report = MBOQualityReport(
            symbol=symbol,
            date=event_date.isoformat(),
            rows=accumulator.rows,
            rth_rows=accumulator.rth_rows,
            action_counts=dict(sorted(accumulator.action_counts.items())),
            duplicate_adds=accumulator.duplicate_adds,
            missing_order_events=accumulator.missing_order_events,
            over_cancels=accumulator.over_cancels,
            reference_mismatches=accumulator.reference_mismatches,
            crossed_count=accumulator.crossed_count,
            outside_rth_crossed_count=accumulator.outside_rth_crossed_count,
            resets=accumulator.resets,
        )
        quality_path = destination / f"{event_date.isoformat()}.quality.json"
        quality_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
        reports.append(report)
        quality_paths.append(quality_path)
    return MBOReconstructionResult(
        quality_paths=tuple(quality_paths),
        reports=tuple(reports),
        final_order_count=len(book.orders),
        runtime_seconds=time.perf_counter() - started,
    )


def _timestamp_ns(value: object) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.value)


class MBOReplayer:
    """Replay-backed exact queue-position calculations."""

    def __init__(self, source: str | Path, *, symbol: str = "ES.c.0") -> None:
        self.source = Path(source)
        self.symbol = symbol

    def simulate_resting_order(
        self,
        side: str,
        price: float,
        size: int,
        t_submit: object,
    ) -> list[RestingFill]:
        """Insert at queue back and return fills implied by later resting fills."""
        side = normalize_side(side)
        price = float(price)
        size = int(size)
        if size <= 0:
            raise ValueError("size must be positive")
        submit_ns = _timestamp_ns(t_submit)
        book = MBOBook()
        submitted = False
        ahead: dict[int, int] = {}
        remaining = size
        fills: list[RestingFill] = []

        for record in iter_mbo_records(self.source, symbol=self.symbol):
            if not submitted and record.index_ns <= submit_ns:
                book.apply(record)
                continue
            if not submitted:
                ahead = {
                    order_id: book.orders[order_id].size
                    for order_id in book.level_order_ids(side, price)
                }
                submitted = True

            current = book.orders.get(record.order_id)
            if (
                record.action == "F"
                and record.side == side
                and record.price == price
                and record.order_id not in ahead
                and not ahead
            ):
                fill_size = min(remaining, record.size)
                if fill_size > 0:
                    fills.append(
                        RestingFill(
                            time=pd.Timestamp(record.ts_event_ns, unit="ns", tz="UTC"),
                            price=price,
                            size=fill_size,
                        )
                    )
                    remaining -= fill_size
                    if remaining == 0:
                        break

            if record.order_id in ahead and current is not None:
                if record.action == "C":
                    remaining_ahead = ahead[record.order_id] - record.size
                    if remaining_ahead <= 0:
                        del ahead[record.order_id]
                    else:
                        ahead[record.order_id] = remaining_ahead
                elif record.action == "M":
                    loses_priority = (
                        record.side != side
                        or record.price != price
                        or record.size > current.size
                    )
                    if loses_priority or record.size == 0:
                        del ahead[record.order_id]
                    else:
                        ahead[record.order_id] = record.size

            book.apply(record)
            if record.action == "R":
                break
        return fills


def find_mbo_file(data_root: str | Path, symbol: str = "ES.c.0") -> Path:
    """Locate a continuous-contract MBO file in sample or lake layouts."""
    stem = symbol.split(".", maxsplit=1)[0].lower()
    root = Path(data_root)
    candidates = (
        root / "samples" / "databento" / f"{stem}_futures_mbo.parquet",
        root / "futures" / "databento" / f"{stem}_futures_mbo.parquet",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"MBO file not found for {symbol} under {root}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruct an order-level MBO book")
    parser.add_argument("--symbol", default="ES.c.0")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    config = load_config()
    source = args.source or find_mbo_file(config.paths.data_dir, args.symbol)
    cache_dir = args.cache_dir or config.paths.cache_dir
    result = reconstruct_mbo(source, cache_dir, symbol=args.symbol)
    for report in result.reports:
        print(json.dumps(asdict(report), sort_keys=True))
    print(json.dumps({"runtime_seconds": result.runtime_seconds}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
