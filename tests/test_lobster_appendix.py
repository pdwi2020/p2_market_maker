from pathlib import Path

import numpy as np

from p2.bybit_replay import StrategySpec
from p2.lobster_appendix import (
    CALIBRATION_SECONDS,
    INVENTORY_LIMIT,
    LATENCY_MS,
    LOBSTER_APPENDIX_COLUMNS,
    REQUOTE_MS,
    lobster_strategy,
    run_lobster_appendix,
)


def _write_sample(data_dir: Path, symbol: str = "AAPL") -> None:
    prefix = f"{symbol}_2012-06-21_34200000_57600000"
    orderbook = data_dir / f"{prefix}_orderbook_10.csv"
    messages = data_dir / f"{prefix}_message_10.csv"
    orderbook.write_text(
        "\n".join(
            [
                "1000100,1,999900,1",
                "1000200,1,1000000,1",
                "1000100,1,999900,1",
                "1000300,1,1000100,1",
                "1000100,1,999900,1",
                "1000200,1,1000000,1",
                "1000300,1,1000100,1",
                "1000200,1,1000000,1",
            ]
        )
    )
    times = (34200.0, 34201.0, 34202.0, 35999.0, 36000.0, 36000.1, 36000.2, 36001.0)
    prices = (999900, 999800, 1000100, 999700, 999900, 1000200, 1000100, 1000200)
    messages.write_text(
        "\n".join(
            f"{time},4,{idx + 1},2,{price},{1 if idx % 2 == 0 else -1}"
            for idx, (time, price) in enumerate(zip(times, prices, strict=True))
        )
    )


def test_lobster_strategy_uses_touch_and_tick_grid() -> None:
    symmetric = lobster_strategy(
        StrategySpec("symmetric"),
        sigma=0.1,
        A=2.0,
        kappa=1.0,
        horizon_seconds=60.0,
    )
    imbalance = lobster_strategy(
        StrategySpec("glft_imbalance", gamma=0.0001, beta=2.0),
        sigma=0.1,
        A=2.0,
        kappa=1.0,
        horizon_seconds=60.0,
    )

    assert np.allclose(
        symmetric(100.0, 0, 0.0, 99.99, 100.01, 4.0, 2.0),
        (99.99, 100.01),
    )
    neutral_bid, neutral_ask = imbalance(100.0, 0, 0.0, 99.99, 100.01, 3.0, 3.0)
    skewed_bid, skewed_ask = imbalance(100.0, 0, 0.0, 99.99, 100.01, 6.0, 2.0)
    assert skewed_bid >= neutral_bid
    assert skewed_ask >= neutral_ask
    assert np.isclose(skewed_bid / 0.01, round(skewed_bid / 0.01))
    assert np.isclose(skewed_ask / 0.01, round(skewed_ask / 0.01))


def test_appendix_runs_available_symbol_and_records_missing(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    selected = (
        StrategySpec("symmetric"),
        StrategySpec("as", gamma=0.0001),
        StrategySpec("glft", gamma=0.0001),
        StrategySpec("glft_imbalance", gamma=0.0001, beta=1.0),
    )

    result = run_lobster_appendix(
        tmp_path,
        selected,
        cancellation_rule="cancel-from-back",
        symbols=("AAPL", "MSFT"),
    )

    assert result.included_symbols == ("AAPL",)
    assert result.missing_symbols == ("MSFT",)
    assert tuple(result.table.columns) == LOBSTER_APPENDIX_COLUMNS
    assert result.table["strategy"].tolist() == [spec.name for spec in selected]
    assert set(result.table["calibration_seconds"]) == {CALIBRATION_SECONDS}
    assert set(result.table["calibration_event_count"]) == {4}
    assert set(result.table["replay_event_count"]) == {4}
    assert set(result.table["requote_ms"]) == {REQUOTE_MS}
    assert set(result.table["latency_ms"]) == {LATENCY_MS}
    assert result.table["max_abs_inventory"].max() <= INVENTORY_LIMIT
    assert result.table["marketable_fill_count"].sum() == 0
    assert np.allclose(
        result.table["net_pnl"],
        result.table["gross_pnl"]
        + result.table["maker_rebates_pnl"]
        - result.table["taker_fees_pnl"],
    )
