import numpy as np
import pytest

from p2.research_models import (
    apply_imbalance_skew,
    calibrate_arrival_intensity,
    decompose_fills,
    fill_fee,
    glft_quotes,
)


def test_glft_quotes_match_closed_form() -> None:
    mid = 100.0
    inventory = 0.02
    gamma = 0.0005
    sigma = 1.2
    A = 8.0
    kappa = 1.5
    order_size = 0.01
    scaled_risk = gamma * order_size
    c1 = np.log(1.0 + scaled_risk / kappa) / scaled_risk
    c2 = np.sqrt(
        gamma
        / (2.0 * A * order_size * kappa)
        * (1.0 + scaled_risk / kappa) ** (kappa / scaled_risk + 1.0)
    )
    expected_reservation = mid - inventory * sigma * c2
    expected_half_spread = c1 + 0.5 * order_size * sigma * c2

    bid, ask = glft_quotes(
        mid,
        inventory,
        gamma=gamma,
        sigma=sigma,
        A=A,
        kappa=kappa,
        order_size=order_size,
    )

    assert bid == pytest.approx(expected_reservation - expected_half_spread)
    assert ask == pytest.approx(expected_reservation + expected_half_spread)


def test_positive_buy_pressure_raises_quotes() -> None:
    bid, ask = apply_imbalance_skew(99.0, 101.0, 0.75, beta=2.0, tick_size=0.1)

    assert bid == pytest.approx(99.15)
    assert ask == pytest.approx(101.15)


def test_calibration_recovers_simulated_arrival_parameters() -> None:
    rng = np.random.default_rng(20250701)
    expected_A = 12.0
    expected_kappa = 0.8
    observed_seconds = 20_000.0
    trade_count = rng.poisson(expected_A * observed_seconds)
    trade_depths = rng.exponential(1.0 / expected_kappa, size=trade_count)

    A, kappa = calibrate_arrival_intensity(trade_depths, observed_seconds)

    assert A == pytest.approx(expected_A, rel=0.02)
    assert kappa == pytest.approx(expected_kappa, rel=0.02)


def test_fee_is_applied_to_each_fill_notional() -> None:
    fees = [fill_fee(100.0, 0.01, 0.0002), fill_fee(110.0, -0.02, 0.0002)]
    rebate = fill_fee(100.0, 0.01, -0.00005)

    assert sum(fees) == pytest.approx(0.00064)
    assert rebate == pytest.approx(-0.00005)


def test_decomposition_sums_to_net_pnl() -> None:
    result = decompose_fills(
        inventory_changes=[0.01, -0.01],
        fill_prices=[99.0, 103.0],
        fill_midpoints=[100.0, 102.0],
        fee_rates=[0.0002, 0.0002],
        final_midpoint=102.0,
    )

    assert result.realized_spread == pytest.approx(0.02)
    assert result.inventory_revaluation == pytest.approx(0.02)
    assert result.fees == pytest.approx(0.000404)
    assert result.net_pnl == pytest.approx(
        result.realized_spread + result.inventory_revaluation - result.fees
    )
