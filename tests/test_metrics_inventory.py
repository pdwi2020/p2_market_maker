import numpy as np
import pytest

from p2.inventory import adjusted_quotes, hard_limit_active
from p2.metrics import average_abs_inventory, fill_rate, inventory_variance, pnl_sharpe, spread_capture


def test_inventory_hard_limit_activation() -> None:
    q = np.array([-2, -1, 0, 1, 2])
    assert np.array_equal(hard_limit_active(q, Q_max=2, direction=1), np.array([False, False, False, False, True]))
    assert np.array_equal(hard_limit_active(q, Q_max=2, direction=-1), np.array([True, False, False, False, False]))


def test_adjusted_quotes_shift_against_inventory() -> None:
    bid, ask = adjusted_quotes(99.0, 101.0, q=2, Q_max=4, skew_factor=0.5)
    assert bid < 99.0
    assert ask < 101.0


def test_metric_helpers() -> None:
    pnl = np.array([1.0, 2.0, 3.0, 4.0])
    inventory = np.array([[0, 1, 2], [0, -1, -2]])
    bid_quotes = np.array([[99.0, 99.0], [99.0, 99.0]])
    ask_quotes = np.array([[101.0, 101.0], [101.0, 101.0]])
    fills = np.array([[1, 0], [0, 1]])

    assert pnl_sharpe(pnl) > 0
    assert inventory_variance(inventory) == pytest.approx(np.var(inventory))
    assert average_abs_inventory(inventory) == pytest.approx(np.mean(np.abs(inventory)))
    assert fill_rate(np.array([2, 2]), n_steps=10, dt=0.1) == pytest.approx(2.0)
    assert spread_capture(bid_quotes, ask_quotes, fills, fills) > 0
