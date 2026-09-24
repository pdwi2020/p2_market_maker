from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from p2.bybit_replay import (
    ReplaySettings,
    _new_working_order,
    StrategySpec,
    load_bybit_events,
    replay_bybit_day,
)
from p2.research_models import DailyCalibration


BASE_MS = 1_751_328_000_000
CALIBRATION = DailyCalibration(A=8.0, kappa=1.5, sigma=1.0, valid_seconds=86_000, trade_count=100)


def _write_stream(
    path: Path,
    *,
    bid_sizes: tuple[float, ...],
    trade_volume: float,
    trade_price: float = 100.0,
    final_bid: float = 100.5,
    final_ask: float = 101.5,
    extra_bid_levels: tuple[tuple[float, float], ...] = (),
    extra_ask_levels: tuple[tuple[float, float], ...] = (),
) -> None:
    book_count = len(bid_sizes) + 1
    times: list[int] = []
    event_types: list[str] = []
    book_valid: list[bool | None] = []
    best_bids: list[float | None] = []
    best_bid_sizes: list[float | None] = []
    best_asks: list[float | None] = []
    best_ask_sizes: list[float | None] = []
    bid_prices: list[list[float] | None] = []
    all_bid_sizes: list[list[float] | None] = []
    ask_prices: list[list[float] | None] = []
    ask_sizes: list[list[float] | None] = []
    trade_prices: list[float | None] = []
    trade_volumes: list[float | None] = []
    trade_sides: list[str | None] = []
    for index, depth in enumerate(bid_sizes):
        times.append(BASE_MS + index * 10)
        event_types.append("book")
        book_valid.append(True)
        best_bids.append(100.0)
        best_bid_sizes.append(depth)
        best_asks.append(101.0)
        best_ask_sizes.append(2.0)
        bid_prices.append([100.0, 99.9, *(price for price, _ in extra_bid_levels)])
        all_bid_sizes.append([depth, 2.0, *(size for _, size in extra_bid_levels)])
        ask_prices.append([101.0, 101.1, *(price for price, _ in extra_ask_levels)])
        ask_sizes.append([2.0, 2.0, *(size for _, size in extra_ask_levels)])
        trade_prices.append(None)
        trade_volumes.append(None)
        trade_sides.append(None)
    times.append(BASE_MS + len(bid_sizes) * 10)
    event_types.append("trade")
    book_valid.append(None)
    best_bids.append(None)
    best_bid_sizes.append(None)
    best_asks.append(None)
    best_ask_sizes.append(None)
    bid_prices.append(None)
    all_bid_sizes.append(None)
    ask_prices.append(None)
    ask_sizes.append(None)
    trade_prices.append(trade_price)
    trade_volumes.append(trade_volume)
    trade_sides.append("Sell")
    times.append(BASE_MS + book_count * 10)
    event_types.append("book")
    book_valid.append(True)
    best_bids.append(final_bid)
    best_bid_sizes.append(1.0)
    best_asks.append(final_ask)
    best_ask_sizes.append(1.0)
    bid_prices.append([final_bid, 100.0])
    all_bid_sizes.append([1.0, 0.0])
    ask_prices.append([final_ask, final_ask + 0.1])
    ask_sizes.append([1.0, 1.0])
    trade_prices.append(None)
    trade_volumes.append(None)
    trade_sides.append(None)
    table = pa.table(
        {
            "time": pa.array(times, pa.timestamp("ms", tz="UTC")),
            "event_type": event_types,
            "book_valid": book_valid,
            "best_bid": best_bids,
            "best_bid_size": best_bid_sizes,
            "best_ask": best_asks,
            "best_ask_size": best_ask_sizes,
            "bid_prices": bid_prices,
            "bid_sizes": all_bid_sizes,
            "ask_prices": ask_prices,
            "ask_sizes": ask_sizes,
            "trade_price": trade_prices,
            "trade_volume": trade_volumes,
            "trade_side": trade_sides,
        }
    )
    pq.write_table(table, path)


def test_symmetric_replay_tracks_queue_fee_and_inventory(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    _write_stream(source, bid_sizes=(1.0,), trade_volume=1.01)

    result = replay_bybit_day(
        source,
        symbol="BTCUSDT",
        date="2025-07-01",
        strategy=StrategySpec("symmetric"),
        calibration=CALIBRATION,
        settings=ReplaySettings(latency_ms=0),
    )

    assert result.fill_count == 1
    assert result.filled_volume == pytest.approx(0.01)
    assert result.end_inventory == pytest.approx(0.01)
    assert result.max_abs_inventory == pytest.approx(0.01)
    assert result.quote_count == 2
    assert result.fees == pytest.approx(0.0002)
    assert result.gross_pnl == pytest.approx(0.01)
    assert result.net_pnl == pytest.approx(0.0098)
    assert result.decomposition.net_pnl == pytest.approx(result.net_pnl)


def test_loaded_events_can_be_reused_across_replays(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    _write_stream(source, bid_sizes=(1.0,), trade_volume=1.01)
    events = load_bybit_events(source)

    first = replay_bybit_day(
        events,
        symbol="BTCUSDT",
        date="2025-07-01",
        strategy=StrategySpec("symmetric"),
        calibration=CALIBRATION,
        settings=ReplaySettings(latency_ms=0),
    )
    second = replay_bybit_day(
        events,
        symbol="BTCUSDT",
        date="2025-07-01",
        strategy=StrategySpec("symmetric"),
        calibration=CALIBRATION,
        settings=ReplaySettings(latency_ms=0),
    )

    assert first == second


def test_cancellation_rules_change_queue_advance(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    _write_stream(source, bid_sizes=(10.0, 15.0, 10.0), trade_volume=7.0)

    back = replay_bybit_day(
        source,
        symbol="BTCUSDT",
        date="2025-07-01",
        strategy=StrategySpec("symmetric"),
        calibration=CALIBRATION,
        settings=ReplaySettings(latency_ms=0, cancellation_rule="cancel-from-back"),
    )
    proportional = replay_bybit_day(
        source,
        symbol="BTCUSDT",
        date="2025-07-01",
        strategy=StrategySpec("symmetric"),
        calibration=CALIBRATION,
        settings=ReplaySettings(latency_ms=0, cancellation_rule="proportional"),
    )

    assert back.fill_count == 0
    assert proportional.fill_count == 1
    assert proportional.filled_volume == pytest.approx(0.01)


def test_inventory_limit_suppresses_risk_increasing_quote(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    _write_stream(source, bid_sizes=(0.0,), trade_volume=0.01)

    result = replay_bybit_day(
        source,
        symbol="BTCUSDT",
        date="2025-07-01",
        strategy=StrategySpec("symmetric"),
        calibration=CALIBRATION,
        settings=ReplaySettings(order_size=0.01, inventory_limit=0.01, latency_ms=0),
    )

    assert result.end_inventory == pytest.approx(0.01)
    assert result.max_abs_inventory <= 0.01 + 1e-12


def _replay(source: Path, strategy: StrategySpec):
    return replay_bybit_day(
        source,
        symbol="BTCUSDT",
        date="2025-07-01",
        strategy=strategy,
        calibration=CALIBRATION,
        settings=ReplaySettings(latency_ms=0),
    )


def test_sweep_does_not_fill_through_a_deep_queue(tmp_path: Path) -> None:
    """A tick through our price is not a licence to fill the whole order.

    Bybit prints one row per matched price level, so a 0.01 BTC print cannot
    have cleared the 100 BTC resting ahead of us at that level.
    """
    source = tmp_path / "events.parquet"
    _write_stream(source, bid_sizes=(100.0,), trade_volume=0.01, trade_price=99.9)

    result = _replay(source, StrategySpec("symmetric"))

    assert result.fill_count == 0


def test_sweep_fills_only_the_volume_beyond_the_queue(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    _write_stream(source, bid_sizes=(1.0,), trade_volume=1.005, trade_price=99.9)

    result = _replay(source, StrategySpec("symmetric"))

    assert result.fill_count == 1
    assert result.filled_volume == pytest.approx(0.005)
    assert result.fills[0].price == pytest.approx(100.0)


def test_quote_alone_on_an_empty_level_fills_from_a_sweep(tmp_path: Path) -> None:
    """An empty tick inside the window leaves us first in line, not unseen."""
    source = tmp_path / "events.parquet"
    _write_stream(
        source,
        bid_sizes=(1.0,),
        trade_volume=0.004,
        trade_price=99.7,
        extra_bid_levels=((99.7, 2.0),),
        extra_ask_levels=((101.2, 2.0),),
    )

    result = _replay(source, StrategySpec("glft", gamma=0.001))

    assert result.beyond_book_placements == 0
    assert result.empty_level_placements >= 1
    assert result.fill_count == 1
    assert result.filled_volume == pytest.approx(0.004)
    assert result.fills[0].price == pytest.approx(99.8)


def test_quote_below_the_reconstructed_book_never_fills(tmp_path: Path) -> None:
    """Without a visible queue we decline to invent one."""
    source = tmp_path / "events.parquet"
    _write_stream(source, bid_sizes=(1.0,), trade_volume=5.0, trade_price=99.7)

    result = _replay(source, StrategySpec("glft", gamma=0.001))

    assert result.beyond_book_placements >= 1
    assert result.fill_count == 0


@pytest.mark.parametrize(
    ("side", "price", "expected"),
    [
        ("bid", 100.0, "displayed"),
        ("bid", 99.95, "empty_level"),
        ("bid", 99.8, "beyond_book"),
        ("ask", 101.0, "displayed"),
        ("ask", 101.05, "empty_level"),
        ("ask", 101.2, "beyond_book"),
    ],
)
def test_placement_separates_empty_levels_from_unseen_depth(
    side: str, price: float, expected: str
) -> None:
    prices = np.array([100.0, 99.9]) if side == "bid" else np.array([101.0, 101.1])
    sizes = np.array([3.0, 2.0])

    order = _new_working_order(side, price, prices, sizes, ReplaySettings())

    assert order.placement == expected
    assert order.fillable is (expected != "beyond_book")
    assert order.ahead == pytest.approx(3.0 if expected == "displayed" else 0.0)
