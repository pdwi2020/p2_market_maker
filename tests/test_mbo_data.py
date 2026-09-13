from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from p2.data.mbo import (
    LAST_EVENT_FLAG,
    MBOBook,
    MBORecord,
    MBOReplayer,
    find_mbo_file,
    is_rth,
    iter_mbo_records,
    reconstruct_mbo,
)


BASE = pd.Timestamp("2025-01-06 14:30:00", tz="UTC")


def _record(
    offset_ns: int,
    action: str,
    side: str = "N",
    price: float = 0.0,
    size: int = 0,
    order_id: int = 0,
    flags: int = LAST_EVENT_FLAG,
) -> MBORecord:
    return MBORecord(BASE.value + offset_ns, action, side, price, size, order_id, flags)


def _write_mbo_fixture(path: Path, *, stale_first_event: bool = False) -> None:
    records = [
        _record(0, "R"),
        _record(1, "A", "B", 100.0, 2, 1),
        _record(2, "A", "B", 100.0, 1, 2),
        _record(3, "A", "A", 101.0, 5, 10),
        _record(5, "F", "B", 100.0, 2, 1, 0),
        _record(6, "C", "B", 100.0, 2, 1),
        _record(7, "C", "B", 100.0, 1, 2),
        _record(8, "A", "B", 100.0, 3, 3),
        _record(9, "F", "B", 100.0, 1, 3, 0),
        _record(10, "C", "B", 100.0, 1, 3),
        _record(11, "F", "B", 100.0, 1, 3, 0),
        _record(12, "C", "B", 100.0, 1, 3),
    ]
    event_times = [record.ts_event_ns for record in records]
    if stale_first_event:
        event_times[0] -= 7_200_000_000_000
    table = pa.table(
        {
            "ts_event": pa.array(
                event_times, pa.timestamp("ns", tz="UTC")
            ),
            "ts_recv": pa.array(
                [record.index_ns for record in records], pa.timestamp("ns", tz="UTC")
            ),
            "action": [record.action for record in records],
            "side": [record.side for record in records],
            "price": [record.price for record in records],
            "size": [record.size for record in records],
            "order_id": [record.order_id for record in records],
            "flags": [record.flags for record in records],
            "symbol": ["ES.c.0"] * len(records),
        }
    )
    pq.write_table(table, path)


def test_book_tracks_cancels_modifications_and_fifo_priority() -> None:
    book = MBOBook()
    book.apply(_record(0, "A", "B", 100.0, 5, 1))
    book.apply(_record(1, "A", "B", 100.0, 3, 2))
    book.apply(_record(2, "A", "B", 100.0, 1, 3))
    book.apply(_record(3, "A", "A", 101.0, 4, 9))

    book.apply(_record(4, "C", "B", 100.0, 2, 1))
    book.apply(_record(5, "M", "B", 100.0, 4, 2))
    assert book.orders[1].size == 3
    assert book.level_order_ids("B", 100.0) == (1, 3, 2)
    assert book.best_bid == 100.0
    assert book.best_ask == 101.0
    assert not book.crossed

    book.apply(_record(6, "M", "B", 100.0, 2, 1))
    assert book.level_order_ids("bid", 100.0) == (1, 3, 2)
    book.apply(_record(7, "M", "B", 99.5, 2, 1))
    assert book.level_order_ids("B", 100.0) == (3, 2)
    assert book.level_order_ids("B", 99.5) == (1,)
    book.apply(_record(8, "R"))
    assert not book.orders


def test_simulated_order_advances_after_fills_and_cancels(tmp_path: Path) -> None:
    source = tmp_path / "mbo.parquet"
    _write_mbo_fixture(source)

    fills = MBOReplayer(source).simulate_resting_order("buy", 100.0, 2, BASE.value + 3)

    assert [fill.size for fill in fills] == [1, 1]
    assert [fill.time.value for fill in fills] == [BASE.value + 9, BASE.value + 11]
    assert all(fill.price == 100.0 for fill in fills)


def test_reconstruction_writes_quality_report(tmp_path: Path) -> None:
    source = tmp_path / "mbo.parquet"
    _write_mbo_fixture(source)

    result = reconstruct_mbo(source, tmp_path / "cache")

    assert len(result.reports) == 1
    report = result.reports[0]
    assert report.date == "2025-01-06"
    assert report.rows == 12
    assert report.rth_rows == 12
    assert report.action_counts == {"A": 4, "C": 4, "F": 3, "R": 1}
    assert report.crossed_count == 0
    assert report.missing_order_events == 0
    assert result.quality_paths[0].is_file()


def test_january_rth_boundaries() -> None:
    assert is_rth(pd.Timestamp("2025-01-06 14:30:00", tz="UTC").value)
    assert is_rth(pd.Timestamp("2025-01-06 20:59:59.999999999", tz="UTC").value)
    assert not is_rth(pd.Timestamp("2025-01-06 21:00:00", tz="UTC").value)
    assert not is_rth(pd.Timestamp("2025-02-06 14:30:00", tz="UTC").value)


def test_rth_stream_uses_receive_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "mbo.parquet"
    _write_mbo_fixture(source, stale_first_event=True)

    records = list(iter_mbo_records(source, rth_only=True))

    assert len(records) == 12
    assert records[0].index_ns == BASE.value
    assert records[0].ts_event_ns < records[0].index_ns


def test_finds_sample_file(tmp_path: Path) -> None:
    source = tmp_path / "samples" / "databento" / "es_futures_mbo.parquet"
    source.parent.mkdir(parents=True)
    source.touch()

    assert find_mbo_file(tmp_path) == source
