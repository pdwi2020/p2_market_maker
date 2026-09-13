from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from p2.data.mbo import LAST_EVENT_FLAG
from p2.queue_validation import METHODS, validate_es_queues


BASE = pd.Timestamp("2025-01-06 14:30:00", tz="UTC")


def _write_validation_fixture(path: Path) -> None:
    rows = [
        (0, "R", "N", 0.0, 0, 0),
        (1, "A", "B", 100.0, 1, 1),
        (2, "A", "A", 101.0, 1, 10),
        (20_000_000, "F", "B", 100.0, 1, 1),
        (21_000_000, "C", "B", 100.0, 1, 1),
        (25_000_000, "A", "B", 100.0, 1, 2),
        (30_000_000, "F", "B", 100.0, 1, 2),
        (31_000_000, "C", "B", 100.0, 1, 2),
        (32_000_000, "A", "B", 100.0, 1, 3),
        (61_000_000_000, "N", "N", 0.0, 0, 0),
    ]
    times = [BASE.value + row[0] for row in rows]
    table = pa.table(
        {
            "ts_event": pa.array(times, pa.timestamp("ns", tz="UTC")),
            "ts_recv": pa.array(times, pa.timestamp("ns", tz="UTC")),
            "action": [row[1] for row in rows],
            "side": [row[2] for row in rows],
            "price": [row[3] for row in rows],
            "size": [row[4] for row in rows],
            "order_id": [row[5] for row in rows],
            "flags": [LAST_EVENT_FLAG] * len(rows),
            "symbol": ["ES.c.0"] * len(rows),
        }
    )
    pq.write_table(table, path)


def test_single_pass_validation_reports_all_methods(tmp_path: Path) -> None:
    source = tmp_path / "es.parquet"
    _write_validation_fixture(source)

    result = validate_es_queues(source, dates=("2025-01-06",))

    aggregate = result.table[result.table["date"] == "ALL"]
    assert set(aggregate["method"]) == set(METHODS)
    assert set(aggregate["fill_count"]) == {1}
    assert set(aggregate["gross_pnl"]) == {25.0}
    assert result.selected_rule == "proportional"
    assert aggregate.loc[
        aggregate["method"] == "l2_proportional", "selected_rule"
    ].item()
