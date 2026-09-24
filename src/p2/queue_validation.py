"""Single-pass ES order-level and level-two queue validation."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from p2.bybit_replay import CancellationRule
from p2.data.mbo import (
    DAY_NS,
    LAST_EVENT_FLAG,
    RTH_END_NS,
    RTH_START_NS,
    MBOBook,
    MBORecord,
    iter_mbo_records,
    normalize_side,
)


VALIDATION_DATES = ("2025-01-06", "2025-01-07")
MARKOUT_HORIZONS = (1, 5, 30, 60)
ES_TICK_SIZE = 0.25
ES_POINT_VALUE = 50.0
METHODS = ("exact_fifo", "l2_cancel_from_back", "l2_proportional")


@dataclass(frozen=True, slots=True)
class _Arm:
    """One quoting configuration evaluated by every queue method."""

    name: str
    offset_ticks: int
    latency_ms: int


# The first arm is the preregistered one and is the only arm the cancellation
# rule is selected on. The others exist because the crypto study quotes away
# from the touch at 50 ms, and a validation that never leaves the touch says
# nothing about the paths that study actually uses.
ARMS = (
    _Arm("touch_10ms", 0, 10),
    _Arm("touch_50ms", 0, 50),
    _Arm("off_2_ticks_50ms", 2, 50),
    _Arm("off_8_ticks_50ms", 8, 50),
)
SELECTION_ARM = ARMS[0].name


@dataclass(frozen=True, slots=True)
class QueueValidationFill:
    """One counterfactual ES fill."""

    time_ns: int
    activated_ns: int
    side: Literal["B", "A"]
    price: float
    midpoint: float

    @property
    def inventory_change(self) -> int:
        return 1 if self.side == "B" else -1


@dataclass(slots=True)
class _ExactOrder:
    side: Literal["B", "A"]
    price: float
    activated_ns: int
    ahead: dict[int, int]


@dataclass(slots=True)
class _LevelOrder:
    side: Literal["B", "A"]
    price: float
    activated_ns: int
    ahead: float
    market_size: float


@dataclass(frozen=True, slots=True)
class _QuoteCommand:
    effective_ns: int
    bid: float | None
    ask: float | None


@dataclass(slots=True)
class _MethodState:
    method: str
    cancellation_rule: CancellationRule | None
    arm: str = SELECTION_ARM
    offset_ticks: int = 0
    latency_ns: int = 10_000_000
    commands: deque[_QuoteCommand] = field(default_factory=deque)
    bid_order: _ExactOrder | _LevelOrder | None = None
    ask_order: _ExactOrder | _LevelOrder | None = None
    submitted_bid: float | None = None
    submitted_ask: float | None = None
    last_decision_ns: int = -(10**30)
    inventory: int = 0
    cash: float = 0.0
    fills: list[QueueValidationFill] = field(default_factory=list)
    last_midpoint: float | None = None
    pending_markouts: list[tuple[int, int, int]] = field(default_factory=list)
    markouts: dict[int, list[float]] = field(
        default_factory=lambda: {horizon: [] for horizon in MARKOUT_HORIZONS}
    )


@dataclass(frozen=True)
class _MethodResult:
    date: str
    arm: str
    offset_ticks: int
    latency_ms: int
    method: str
    fills: tuple[QueueValidationFill, ...]
    gross_pnl: float
    markouts: dict[int, tuple[float, ...]]


@dataclass(frozen=True)
class QueueValidationResult:
    """Published validation rows and the selected crypto cancellation rule."""

    table: pd.DataFrame
    selected_rule: CancellationRule


def _level_size(book: MBOBook, side: str, price: float) -> float:
    return float(
        sum(
            book.orders[order_id].size
            for order_id in book.level_order_ids(side, price)
        )
    )


def _new_order(
    state: _MethodState,
    side: Literal["B", "A"],
    price: float,
    effective_ns: int,
    book: MBOBook,
) -> _ExactOrder | _LevelOrder:
    if state.method == "exact_fifo":
        ahead = {
            order_id: book.orders[order_id].size
            for order_id in book.level_order_ids(side, price)
        }
        return _ExactOrder(side, price, effective_ns, ahead)
    depth = _level_size(book, side, price)
    return _LevelOrder(side, price, effective_ns, depth, depth)


def _activate_due(state: _MethodState, now_ns: int, book: MBOBook) -> None:
    while state.commands and state.commands[0].effective_ns <= now_ns:
        command = state.commands.popleft()
        bid = command.bid if state.inventory < 5 else None
        ask = command.ask if state.inventory > -5 else None
        if book.best_bid is None or book.best_ask is None:
            state.bid_order = None
            state.ask_order = None
            continue
        if bid is not None and bid >= book.best_ask:
            bid = book.best_ask - ES_TICK_SIZE
        if ask is not None and ask <= book.best_bid:
            ask = book.best_bid + ES_TICK_SIZE
        if bid is None:
            state.bid_order = None
        elif state.bid_order is None or state.bid_order.price != bid:
            state.bid_order = _new_order(
                state, "B", bid, command.effective_ns, book
            )
        if ask is None:
            state.ask_order = None
        elif state.ask_order is None or state.ask_order.price != ask:
            state.ask_order = _new_order(
                state, "A", ask, command.effective_ns, book
            )


def _resolve_markouts(state: _MethodState, now_ns: int) -> None:
    while state.pending_markouts and state.pending_markouts[0][0] <= now_ns:
        _, fill_index, horizon = heapq.heappop(state.pending_markouts)
        if state.last_midpoint is None:
            continue
        fill = state.fills[fill_index]
        value = (
            fill.inventory_change
            * (state.last_midpoint - fill.midpoint)
            * ES_POINT_VALUE
        )
        state.markouts[horizon].append(float(value))


def _record_fill(
    state: _MethodState,
    order: _ExactOrder | _LevelOrder,
    now_ns: int,
) -> None:
    if state.last_midpoint is None:
        return
    fill = QueueValidationFill(
        time_ns=now_ns,
        activated_ns=order.activated_ns,
        side=order.side,
        price=order.price,
        midpoint=state.last_midpoint,
    )
    state.fills.append(fill)
    state.inventory += fill.inventory_change
    state.cash -= fill.inventory_change * fill.price * ES_POINT_VALUE
    fill_index = len(state.fills) - 1
    for horizon in MARKOUT_HORIZONS:
        heapq.heappush(
            state.pending_markouts,
            (now_ns + horizon * 1_000_000_000, fill_index, horizon),
        )
    if order.side == "B":
        state.bid_order = None
        state.submitted_bid = None
    else:
        state.ask_order = None
        state.submitted_ask = None


def _consume_ahead(order: _ExactOrder | _LevelOrder, volume: float) -> float:
    """Charge aggressor volume to the queue ahead and return what passes us."""
    if isinstance(order, _ExactOrder):
        residual = volume
        for order_id in list(order.ahead):
            taken = min(float(order.ahead[order_id]), residual)
            remaining = order.ahead[order_id] - taken
            if remaining <= 0:
                del order.ahead[order_id]
            else:
                order.ahead[order_id] = int(remaining)
            residual -= taken
            if residual <= 0.0:
                return 0.0
        return residual
    consumed = min(order.ahead, volume)
    order.ahead -= consumed
    return volume - consumed


def _process_fill_record(state: _MethodState, record: MBORecord) -> None:
    """Apply the same at-or-through volume rule the crypto replay uses.

    A fill printing past our price can only have reached there by clearing the
    queue ahead of us, and it can only fill what it actually traded. Handling
    the two cases alike is what lets these rows speak to the crypto engine
    rather than to a rule only this module implements.
    """
    if record.action != "F":
        return
    side = normalize_side(record.side)
    order = state.bid_order if side == "B" else state.ask_order
    if order is None:
        return
    at_price = record.price == order.price
    through = (
        order.side == "B" and record.price < order.price
    ) or (order.side == "A" and record.price > order.price)
    if not (at_price or through):
        return
    if isinstance(order, _ExactOrder) and record.order_id in order.ahead:
        # This maker sits ahead of us; its own cancel record drains the queue.
        return
    residual = _consume_ahead(order, float(record.size))
    if isinstance(order, _LevelOrder):
        order.market_size = (
            0.0 if through else max(order.market_size - record.size, 0.0)
        )
    if residual >= 1.0:
        _record_fill(state, order, record.index_ns)


def _update_exact_ahead(
    state: _MethodState,
    record: MBORecord,
    current_side: str | None,
    current_price: float | None,
    current_size: int | None,
) -> None:
    for order in (state.bid_order, state.ask_order):
        if not isinstance(order, _ExactOrder) or record.order_id not in order.ahead:
            continue
        if record.action == "C":
            remaining = order.ahead[record.order_id] - record.size
            if remaining <= 0:
                del order.ahead[record.order_id]
            else:
                order.ahead[record.order_id] = remaining
        elif record.action == "M" and current_size is not None:
            loses_priority = (
                record.side != current_side
                or record.price != current_price
                or record.size > current_size
            )
            if loses_priority or record.size == 0:
                del order.ahead[record.order_id]
            else:
                order.ahead[record.order_id] = record.size


def _affected_levels(
    record: MBORecord,
    current_side: str | None,
    current_price: float | None,
) -> set[tuple[str, float]]:
    levels: set[tuple[str, float]] = set()
    if current_side is not None and current_price is not None:
        levels.add((current_side, current_price))
    if record.action in {"A", "M"}:
        levels.add((normalize_side(record.side), float(record.price)))
    return levels


def _update_level_orders(
    state: _MethodState,
    affected: set[tuple[str, float]],
    book: MBOBook,
) -> None:
    for order in (state.bid_order, state.ask_order):
        if not isinstance(order, _LevelOrder):
            continue
        if (order.side, order.price) not in affected:
            continue
        new_size = _level_size(book, order.side, order.price)
        reduction = max(order.market_size - new_size, 0.0)
        if reduction > 0.0 and order.ahead > 0.0:
            if state.cancellation_rule == "cancel-from-back":
                behind = max(order.market_size - order.ahead, 0.0)
                order.ahead = max(order.ahead - max(reduction - behind, 0.0), 0.0)
            elif order.market_size > 0.0:
                order.ahead *= max(1.0 - reduction / order.market_size, 0.0)
        order.market_size = new_size


def _submit_quotes(state: _MethodState, now_ns: int, book: MBOBook) -> None:
    if now_ns - state.last_decision_ns < 100_000_000:
        return
    if state.offset_ticks == 0:
        raw_bid, raw_ask = book.best_bid, book.best_ask
    else:
        midpoint = 0.5 * (book.best_bid + book.best_ask)
        offset = state.offset_ticks * ES_TICK_SIZE
        raw_bid = float(
            np.floor((midpoint - offset) / ES_TICK_SIZE + 1e-9) * ES_TICK_SIZE
        )
        raw_ask = float(
            np.ceil((midpoint + offset) / ES_TICK_SIZE - 1e-9) * ES_TICK_SIZE
        )
    bid = raw_bid if state.inventory < 5 else None
    ask = raw_ask if state.inventory > -5 else None
    if bid != state.submitted_bid or ask != state.submitted_ask:
        state.commands.append(_QuoteCommand(now_ns + state.latency_ns, bid, ask))
        state.submitted_bid = bid
        state.submitted_ask = ask
    state.last_decision_ns = now_ns


_RULES: dict[str, CancellationRule | None] = {
    "exact_fifo": None,
    "l2_cancel_from_back": "cancel-from-back",
    "l2_proportional": "proportional",
}


def _new_states() -> dict[tuple[str, str], _MethodState]:
    return {
        (arm.name, method): _MethodState(
            method,
            _RULES[method],
            arm=arm.name,
            offset_ticks=arm.offset_ticks,
            latency_ns=arm.latency_ms * 1_000_000,
        )
        for arm in ARMS
        for method in METHODS
    }


def _finish_day(
    day: str,
    states: dict[tuple[str, str], _MethodState],
    end_ns: int,
) -> list[_MethodResult]:
    output: list[_MethodResult] = []
    for state in states.values():
        _resolve_markouts(state, end_ns)
        if state.last_midpoint is None:
            raise ValueError(f"no valid ES midpoint is available for {day}")
        gross_pnl = (
            state.cash + state.inventory * state.last_midpoint * ES_POINT_VALUE
        )
        output.append(
            _MethodResult(
                date=day,
                arm=state.arm,
                offset_ticks=state.offset_ticks,
                latency_ms=state.latency_ns // 1_000_000,
                method=state.method,
                fills=tuple(state.fills),
                gross_pnl=float(gross_pnl),
                markouts={
                    horizon: tuple(values)
                    for horizon, values in state.markouts.items()
                },
            )
        )
    return output


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return np.nan
    points = np.sort(np.concatenate([left, right]))
    left_cdf = np.searchsorted(np.sort(left), points, side="right") / len(left)
    right_cdf = np.searchsorted(np.sort(right), points, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _combine(
    results: Sequence[_MethodResult], arm: str, method: str
) -> _MethodResult:
    selected = [
        result
        for result in results
        if result.method == method and result.arm == arm
    ]
    template = selected[0]
    return _MethodResult(
        date="ALL",
        arm=arm,
        offset_ticks=template.offset_ticks,
        latency_ms=template.latency_ms,
        method=method,
        fills=tuple(fill for result in selected for fill in result.fills),
        gross_pnl=float(sum(result.gross_pnl for result in selected)),
        markouts={
            horizon: tuple(
                value
                for result in selected
                for value in result.markouts[horizon]
            )
            for horizon in MARKOUT_HORIZONS
        },
    )


def _validation_rows(results: Sequence[_MethodResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes = sorted({result.date for result in results}) + ["ALL"]
    for arm in ARMS:
        for scope in scopes:
            rows.extend(_arm_rows(results, arm.name, scope))
    return rows


def _arm_rows(
    results: Sequence[_MethodResult], arm: str, scope: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scoped = (
        [_combine(results, arm, method) for method in METHODS]
        if scope == "ALL"
        else [
            result
            for result in results
            if result.date == scope and result.arm == arm
        ]
    )
    exact = next(result for result in scoped if result.method == "exact_fifo")
    exact_times = np.asarray(
        [
            (fill.time_ns - fill.activated_ns) / 1_000_000.0
            for fill in exact.fills
        ]
    )
    for result in scoped:
        fill_times = np.asarray(
            [
                (fill.time_ns - fill.activated_ns) / 1_000_000.0
                for fill in result.fills
            ]
        )
        if result.method == "exact_fifo":
            relative_bias = 0.0
            ks = 0.0
            pnl_error = 0.0
        else:
            relative_bias = (
                (len(result.fills) - len(exact.fills)) / len(exact.fills)
                if exact.fills
                else np.nan
            )
            ks = _ks_distance(fill_times, exact_times)
            pnl_error = result.gross_pnl - exact.gross_pnl
        row: dict[str, object] = {
            "study": "es_queue",
            "symbol": "ES.c.0",
            "date": scope,
            "arm": result.arm,
            "offset_ticks": result.offset_ticks,
            "latency_ms": result.latency_ms,
            "method": result.method,
            "fill_count": len(result.fills),
            "filled_volume": float(len(result.fills)),
            "fill_time_mean_ms": (
                float(np.mean(fill_times)) if len(fill_times) else np.nan
            ),
            "fill_time_median_ms": (
                float(np.median(fill_times)) if len(fill_times) else np.nan
            ),
            "fill_time_p95_ms": (
                float(np.quantile(fill_times, 0.95)) if len(fill_times) else np.nan
            ),
            "fill_count_relative_bias": relative_bias,
            "fill_time_ks": ks,
            "gross_pnl": result.gross_pnl,
            "gross_pnl_error": pnl_error,
        }
        for horizon in MARKOUT_HORIZONS:
            values = result.markouts[horizon]
            row[f"markout_{horizon}s"] = (
                float(np.mean(values)) if values else np.nan
            )
        rows.append(row)
    return rows


def select_cancellation_rule(table: pd.DataFrame) -> CancellationRule:
    """Apply the locked aggregate queue-model selection rule."""
    aggregate = table[
        (table["date"] == "ALL") & (table["arm"] == SELECTION_ARM)
    ]
    candidates = aggregate[aggregate["method"] != "exact_fifo"]
    if len(candidates) != 2:
        raise ValueError("queue validation requires both level-two methods")

    def finite_or_inf(value: float) -> float:
        return abs(float(value)) if np.isfinite(value) else np.inf

    ranked = sorted(
        candidates.to_dict("records"),
        key=lambda row: (
            finite_or_inf(row["fill_count_relative_bias"]),
            finite_or_inf(row["fill_time_ks"]),
            finite_or_inf(row["gross_pnl_error"]),
            0 if row["method"] == "l2_proportional" else 1,
        ),
    )
    return (
        "proportional"
        if ranked[0]["method"] == "l2_proportional"
        else "cancel-from-back"
    )


def validate_es_queues(
    source: str | Path,
    *,
    symbol: str = "ES.c.0",
    dates: Sequence[str] = VALIDATION_DATES,
) -> QueueValidationResult:
    """Evaluate exact and approximate queues together in one MBO pass."""
    selected_days = {
        pd.Timestamp(date.fromisoformat(value), tz="UTC").value // DAY_NS: value
        for value in dates
    }
    book = MBOBook()
    active_day: int | None = None
    states: dict[str, _MethodState] | None = None
    results: list[_MethodResult] = []
    observed_days: set[int] = set()

    for record in iter_mbo_records(source, symbol=symbol):
        now_ns = record.index_ns
        day_number = now_ns // DAY_NS
        time_of_day = now_ns % DAY_NS
        in_selected_rth = (
            day_number in selected_days
            and RTH_START_NS <= time_of_day < RTH_END_NS
        )
        if active_day is not None and not in_selected_rth:
            if states is None:
                raise RuntimeError("queue states were not initialized")
            results.extend(
                _finish_day(
                    selected_days[active_day],
                    states,
                    active_day * DAY_NS + RTH_END_NS,
                )
            )
            active_day = None
            states = None
        if in_selected_rth and active_day is None:
            active_day = day_number
            states = _new_states()
            observed_days.add(day_number)

        if states is not None:
            for state in states.values():
                _resolve_markouts(state, now_ns)
                _activate_due(state, now_ns, book)
                _process_fill_record(state, record)
            current = book.orders.get(record.order_id)
            current_side = current.side if current is not None else None
            current_price = current.price if current is not None else None
            current_size = current.size if current is not None else None
            for state in states.values():
                _update_exact_ahead(
                    state,
                    record,
                    current_side,
                    current_price,
                    current_size,
                )
            affected = _affected_levels(record, current_side, current_price)
        else:
            affected = set()

        book.apply(record)

        if states is not None:
            if record.action == "R":
                for state in states.values():
                    state.commands.clear()
                    state.bid_order = None
                    state.ask_order = None
                    state.submitted_bid = None
                    state.submitted_ask = None
            else:
                for state in states.values():
                    _update_level_orders(state, affected, book)
            stable = bool(record.flags & LAST_EVENT_FLAG)
            if (
                stable
                and book.best_bid is not None
                and book.best_ask is not None
                and not book.crossed
            ):
                midpoint = 0.5 * (book.best_bid + book.best_ask)
                for state in states.values():
                    state.last_midpoint = midpoint
                    _submit_quotes(state, now_ns, book)

    if active_day is not None and states is not None:
        results.extend(
            _finish_day(
                selected_days[active_day],
                states,
                active_day * DAY_NS + RTH_END_NS,
            )
        )
    missing = set(selected_days) - observed_days
    if missing:
        missing_dates = [selected_days[value] for value in sorted(missing)]
        raise ValueError(f"ES validation dates are missing: {missing_dates}")
    table = pd.DataFrame(_validation_rows(results))
    selected_rule = select_cancellation_rule(table)
    table["selected_rule"] = table["method"].eq(
        f"l2_{selected_rule.replace('-', '_')}"
    )
    return QueueValidationResult(table=table, selected_rule=selected_rule)
