import numpy as np
import pytest

from p2.research_metrics import (
    annualized_sharpe,
    deflated_sharpe_ratio,
    max_drawdown,
    moving_block_bootstrap_sharpe,
    summarize_performance,
)


def test_sharpe_and_drawdown() -> None:
    values = [1.0, -2.0, 3.0, -1.0]

    expected_sharpe = np.sqrt(365.0) * np.mean(values) / np.std(values, ddof=1)
    assert annualized_sharpe(values) == pytest.approx(expected_sharpe)
    assert max_drawdown(values) == pytest.approx(2.0)


def test_block_bootstrap_is_seeded_and_ordered() -> None:
    values = np.sin(np.arange(40, dtype=float)) + 0.1

    first = moving_block_bootstrap_sharpe(values, resamples=200, seed=17)
    second = moving_block_bootstrap_sharpe(values, resamples=200, seed=17)

    assert first == second
    assert first[0] < first[1]


def test_deflated_sharpe_uses_declared_trial_count() -> None:
    values = np.linspace(-0.5, 1.0, 60)
    trials = np.linspace(-0.2, 0.3, 31)

    ratio = deflated_sharpe_ratio(values, trials)

    assert 0.0 <= ratio <= 1.0
    with pytest.raises(ValueError, match="trial count"):
        deflated_sharpe_ratio(values, trials[:-1])


def test_performance_summary_contains_all_daily_observations() -> None:
    values = np.sin(np.arange(60, dtype=float)) + 0.2
    trials = np.linspace(-0.1, 0.2, 31)

    summary = summarize_performance(values, trials, bootstrap_resamples=100)

    assert summary.days == 60
    assert summary.total_net_pnl == pytest.approx(sum(values))
    assert summary.max_drawdown >= 0.0
