from types import SimpleNamespace

import numpy as np
import pytest

from p2.bybit_replay import BybitEventArrays
from p2.hft_crosscheck import (
    ExternalReplayResult,
    comparison_row,
    normalize_bybit_events,
)


EVENT_DTYPE = np.dtype(
    [
        ("ev", "<u8"),
        ("exch_ts", "<i8"),
        ("local_ts", "<i8"),
        ("px", "<f8"),
        ("qty", "<f8"),
        ("order_id", "<u8"),
        ("ival", "<i8"),
        ("fval", "<f8"),
    ],
    align=True,
)


def _events() -> BybitEventArrays:
    return BybitEventArrays(
        times=np.asarray([1_000, 1_001, 1_002]),
        is_book=np.asarray([True, True, False]),
        book_times=np.asarray([1_000, 1_001]),
        book_valid=np.asarray([True, True]),
        best_bids=np.asarray([100.0, 100.0]),
        best_bid_sizes=np.asarray([1.0, 2.0]),
        best_asks=np.asarray([101.0, 101.0]),
        best_ask_sizes=np.asarray([1.0, 1.0]),
        bid_offsets=np.asarray([0, 1, 2]),
        bid_prices=np.asarray([100.0, 100.0]),
        bid_sizes=np.asarray([1.0, 2.0]),
        ask_offsets=np.asarray([0, 1, 2]),
        ask_prices=np.asarray([101.0, 101.0]),
        ask_sizes=np.asarray([1.0, 1.0]),
        trade_times=np.asarray([1_002]),
        trade_prices=np.asarray([100.0]),
        trade_volumes=np.asarray([0.01]),
        trade_sides=np.asarray(["Sell"], dtype=object),
    )


def test_normalized_adapter_emits_snapshot_delta_and_trade() -> None:
    bindings = SimpleNamespace(
        event_dtype=EVENT_DTYPE,
        EXCH_EVENT=1 << 31,
        LOCAL_EVENT=1 << 30,
        BUY_EVENT=1 << 29,
        SELL_EVENT=1 << 28,
        DEPTH_EVENT=1,
        TRADE_EVENT=2,
        DEPTH_SNAPSHOT_EVENT=4,
    )

    normalized = normalize_bybit_events(_events(), bindings)

    assert len(normalized) == 4
    assert normalized[0]["ev"] & bindings.DEPTH_SNAPSHOT_EVENT
    assert normalized[2]["ev"] & bindings.DEPTH_EVENT
    assert normalized[2]["qty"] == pytest.approx(2.0)
    assert normalized[3]["ev"] & bindings.TRADE_EVENT
    assert normalized[3]["ev"] & bindings.SELL_EVENT
    assert normalized[3]["exch_ts"] == 1_002_000_000


def test_crosscheck_row_reports_absolute_and_relative_differences() -> None:
    external = ExternalReplayResult(
        fill_count=12,
        filled_volume=0.12,
        gross_pnl=2.0,
        fees=0.5,
        net_pnl=1.5,
        end_inventory=0.0,
    )

    row = comparison_row(
        trading_date="2025-07-07",
        strategy="symmetric",
        native={"fill_count": 10, "filled_volume": 0.1, "net_pnl": 1.0},
        external=external,
        package_version="2.4.4",
    )

    assert row["absolute_difference_fill_count"] == pytest.approx(2.0)
    assert row["relative_difference_fill_count"] == pytest.approx(0.2)
    assert row["absolute_difference_net_pnl"] == pytest.approx(0.5)
    assert row["relative_difference_net_pnl"] == pytest.approx(0.5)
