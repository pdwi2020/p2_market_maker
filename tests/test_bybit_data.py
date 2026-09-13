import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from p2.data.bybit import find_bybit_files, reconstruct_bybit_day


DAY = "2025-06-02"
START_MS = 1_748_822_400_000


def _write_bybit_fixture(root: Path, *, gap: bool = False, crossed: bool = False) -> tuple[Path, Path]:
    orderbook = pa.table(
        {
            "ts": [START_MS + 100, START_MS + 200, START_MS + 300],
            "cts": [START_MS + 100, START_MS + 200, START_MS + 300],
            "type": ["snapshot", "delta", "delta"],
            "update_id": [100, 102 if gap else 101, 103 if gap else 102],
            "bids": [
                '[["100.0", "2.0"], ["99.5", "4.0"]]',
                '[["100.0", "0"], ["100.5", "3.0"]]',
                '[]',
            ],
            "asks": [
                '[["101.0", "5.0"], ["101.5", "1.0"]]',
                '[]',
                '[["101.0", "0"], ["100.4", "1.0"]]' if crossed else '[["101.0", "4.0"]]',
            ],
        }
    )
    trades = pa.table(
        {
            "timestamp": [START_MS + 150, START_MS + 250],
            "price": [101.0, 102.0],
            "volume": [0.25, 0.5],
            "side": ["Buy", "Sell"],
        }
    )
    orderbook_path = root / "orderbook.parquet"
    trades_path = root / "trades.parquet"
    pq.write_table(orderbook, orderbook_path)
    pq.write_table(trades, trades_path)
    return orderbook_path, trades_path


def test_reconstructs_snapshot_delta_and_merged_trades(tmp_path: Path) -> None:
    orderbook_path, trades_path = _write_bybit_fixture(tmp_path)

    result = reconstruct_bybit_day(
        orderbook_path, trades_path, tmp_path / "cache", "BTCUSDT", DAY, top_n=2
    )

    assert result.cache_path == tmp_path / "cache" / "bybit" / "BTCUSDT" / f"{DAY}.parquet"
    assert result.quality.book_events == 3
    assert result.quality.trade_events == 2
    assert result.quality.gaps == 0
    assert result.quality.invalid_book_seconds == pytest.approx(0.1)
    assert result.quality.crossed_count == 0
    assert result.quality.trades_outside_book == 1
    assert result.quality.trades_checked == 2

    cached = pq.read_table(result.cache_path)
    assert cached.num_rows == 5
    assert cached["time"].to_pylist() == sorted(cached["time"].to_pylist())
    books = cached.filter(pa.compute.equal(cached["event_type"], "book"))
    assert books["best_bid"].to_pylist() == [100.0, 100.5, 100.5]
    assert books["best_ask_size"].to_pylist() == [5.0, 5.0, 4.0]
    assert books["bid_prices"].to_pylist()[1] == [100.5, 99.5]
    assert json.loads(result.quality_path.read_text())["crossed_count"] == 0


def test_gap_invalidates_book_until_next_snapshot(tmp_path: Path) -> None:
    orderbook_path, trades_path = _write_bybit_fixture(tmp_path, gap=True)

    result = reconstruct_bybit_day(
        orderbook_path, trades_path, tmp_path / "cache", "BTCUSDT", DAY
    )

    assert result.quality.gaps == 1
    assert result.quality.invalid_book_seconds == pytest.approx(86_399.9)
    cached = pq.read_table(result.cache_path)
    books = cached.filter(pa.compute.equal(cached["event_type"], "book"))
    assert books["book_valid"].to_pylist() == [True, False, False]


def test_crossed_book_is_rejected(tmp_path: Path) -> None:
    orderbook_path, trades_path = _write_bybit_fixture(tmp_path, crossed=True)

    with pytest.raises(ValueError, match="crossed book"):
        reconstruct_bybit_day(
            orderbook_path, trades_path, tmp_path / "cache", "BTCUSDT", DAY
        )


def test_finds_sample_layout(tmp_path: Path) -> None:
    base = tmp_path / "samples" / "bybit" / "BTCUSDT"
    book = base / "orderbook_l2" / f"date={DAY}" / "orderbook.parquet"
    trades = base / "trades" / f"date={DAY}" / "trades.parquet"
    book.parent.mkdir(parents=True)
    trades.parent.mkdir(parents=True)
    book.touch()
    trades.touch()

    assert find_bybit_files(tmp_path, "BTCUSDT", DAY) == (book, trades)
