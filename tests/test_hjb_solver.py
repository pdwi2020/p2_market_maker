import pytest

from p2.hjb_solver import optimal_spread, reservation_price


def test_reservation_price_zero_inventory() -> None:
    assert reservation_price(S=100.0, q=0, t=0.25, T=1.0, gamma=0.1, sigma=1.0) == pytest.approx(100.0)


def test_reservation_price_long() -> None:
    assert reservation_price(S=100.0, q=5, t=0.25, T=1.0, gamma=0.1, sigma=1.0) < 100.0


def test_spread_positive() -> None:
    assert optimal_spread(t=0.0, T=1.0, gamma=0.1, sigma=1.0, kappa=1.5) > 0.0


def test_spread_limit_gamma_zero() -> None:
    spread = optimal_spread(t=0.0, T=1.0, gamma=1e-8, sigma=1.0, kappa=1.5)
    assert spread == pytest.approx(1.0 / 1.5, rel=1e-5)


def test_spread_monotone_sigma() -> None:
    low_sigma = optimal_spread(t=0.0, T=1.0, gamma=0.1, sigma=0.5, kappa=1.5)
    high_sigma = optimal_spread(t=0.0, T=1.0, gamma=0.1, sigma=2.0, kappa=1.5)
    assert high_sigma > low_sigma


def test_spread_monotone_time() -> None:
    early = optimal_spread(t=0.0, T=1.0, gamma=0.1, sigma=1.0, kappa=1.5)
    late = optimal_spread(t=0.9, T=1.0, gamma=0.1, sigma=1.0, kappa=1.5)
    assert early > late
