from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import p2.research_study as research_study
from p2.bybit_replay import (
    CryptoFill,
    DailyReplayResult,
    ReplaySettings,
    StrategySpec,
)
from p2.research_models import PnLDecomposition
from p2.research_study import (
    eligible_dates,
    replay_result_rows,
    select_strategies,
    strategy_candidates,
)


def test_strategy_grid_has_preregistered_31_trials() -> None:
    candidates = strategy_candidates()

    assert len(candidates) == 31
    assert sum(spec.kind == "symmetric" for spec in candidates) == 1
    assert sum(spec.kind == "as" for spec in candidates) == 5
    assert sum(spec.kind == "glft" for spec in candidates) == 5
    assert sum(spec.kind == "glft_imbalance" for spec in candidates) == 20


def test_family_selection_uses_locked_tie_breaking() -> None:
    rows = []
    for spec in strategy_candidates():
        bonus = 1.0
        if spec == StrategySpec("as", gamma=0.00005):
            bonus = 2.0
        if spec.kind == "glft":
            bonus = 3.0
        if spec.kind == "glft_imbalance":
            bonus = 4.0
        rows.extend(
            {"date": f"2025-05-0{index + 1}", "strategy": spec.name, "net_pnl": value}
            for index, value in enumerate((bonus - 1.0, bonus, bonus + 2.0))
        )

    selected = select_strategies(pd.DataFrame(rows))

    assert len(selected.trial_sharpes) == 31
    assert selected.strategies[1] == StrategySpec("as", gamma=0.00005)
    assert selected.strategies[2] == StrategySpec("glft", gamma=0.00001)
    assert selected.strategies[3] == StrategySpec(
        "glft_imbalance", gamma=0.00001, beta=0.5
    )


def test_weekly_dates_start_on_wednesday_and_require_prior_day() -> None:
    available = {
        date(2025, 7, 1),
        date(2025, 7, 2),
        date(2025, 7, 3),
        date(2025, 7, 8),
        date(2025, 7, 10),
    }

    selected = eligible_dates(
        available, date(2025, 7, 1), date(2025, 7, 10), weekly=True
    )

    assert selected == (date(2025, 7, 2),)


def test_fee_grid_is_derived_per_fill_and_reconciles() -> None:
    decomposition = PnLDecomposition(
        realized_spread=0.2,
        inventory_revaluation=0.3,
        fees=0.0,
        net_pnl=0.5,
    )
    result = DailyReplayResult(
        symbol="BTCUSDT",
        date="2025-07-01",
        strategy="symmetric",
        fills=(
            CryptoFill(
                time_ms=1,
                side="bid",
                price=100.0,
                size=0.01,
                midpoint=100.1,
                fee=0.0,
            ),
        ),
        quote_count=2,
        quote_volume=0.02,
        unobserved_placements=0,
        gross_pnl=0.5,
        fees=0.0,
        net_pnl=0.5,
        end_inventory=0.01,
        mean_abs_inventory=0.005,
        inventory_std=0.005,
        max_abs_inventory=0.01,
        final_midpoint=100.5,
        decomposition=decomposition,
        markouts={1: (0.001,), 5: ()},
    )

    daily, decomposed, markouts = replay_result_rows(
        result,
        calibration_date="2025-06-30",
        settings=ReplaySettings(maker_fee_rate=0.0, latency_ms=50),
        maker_fee_rates=(0.0002, -0.00005),
    )

    assert daily[0]["fees"] == pytest.approx(0.0002)
    assert daily[0]["net_pnl"] == pytest.approx(0.4998)
    assert daily[1]["fees"] == pytest.approx(-0.00005)
    assert decomposed[0]["net_pnl"] == pytest.approx(
        decomposed[0]["realized_spread"]
        + decomposed[0]["inventory_revaluation"]
        - decomposed[0]["fee_cost"]
    )
    assert markouts[0]["mean_markout"] == pytest.approx(0.001)
    assert np.isnan(markouts[1]["mean_markout"])


def test_invalid_calibration_dates_are_skipped(
    tmp_path,
    monkeypatch,
) -> None:
    def invalid_calibration(path):
        del path
        raise ValueError("invalid calibration")

    monkeypatch.setattr(
        research_study,
        "calibrate_bybit_day",
        invalid_calibration,
    )
    trading_day = date(2025, 7, 2)

    selection_rows = research_study._selection_day_rows(
        (tmp_path, "proportional", trading_day)
    )
    daily, decomposition, markouts, skipped = research_study._test_day_rows(
        (
            tmp_path,
            "SOLUSDT",
            trading_day,
            (StrategySpec("symmetric"),),
            "proportional",
        )
    )

    assert selection_rows is None
    assert daily == []
    assert decomposition == []
    assert markouts == []
    assert skipped == {
        "symbol": "SOLUSDT",
        "date": "2025-07-02",
        "calibration_date": "2025-07-01",
    }


def test_selection_day_checkpoints_are_reused_and_invalidated(
    tmp_path,
    monkeypatch,
) -> None:
    trading_day = date(2025, 5, 2)
    for day in (trading_day - timedelta(days=1), trading_day):
        path = tmp_path / "bybit" / "BTCUSDT" / f"{day.isoformat()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stream")
    calls = []

    def fake_selection_day_rows(task):
        calls.append(task[2])
        return [
            {
                "date": task[2].isoformat(),
                "strategy": spec.name,
                "net_pnl": float(index),
            }
            for index, spec in enumerate(research_study.strategy_candidates())
        ]

    monkeypatch.setattr(
        research_study,
        "_selection_day_rows",
        fake_selection_day_rows,
    )

    first = research_study.run_selection(
        tmp_path,
        cancellation_rule="proportional",
        selection_dates=(trading_day,),
    )
    second = research_study.run_selection(
        tmp_path,
        cancellation_rule="proportional",
        selection_dates=(trading_day,),
    )

    assert calls == [trading_day]
    assert first.strategies == second.strategies

    current_stream = (
        tmp_path / "bybit" / "BTCUSDT" / f"{trading_day.isoformat()}.parquet"
    )
    current_stream.write_bytes(b"changed stream")
    research_study.run_selection(
        tmp_path,
        cancellation_rule="proportional",
        selection_dates=(trading_day,),
    )

    assert calls == [trading_day, trading_day]
