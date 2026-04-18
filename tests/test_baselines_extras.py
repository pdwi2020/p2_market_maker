from __future__ import annotations

import pytest

from p2.baselines import AvSOptimalMM, InventoryLinearMM, RandomQuoter, SymmetricMM
from p2.hjb_solver import optimal_spread, reservation_price


def test_inventory_linear_collapse() -> None:
    symmetric = SymmetricMM(half_spread=0.4)
    inventory_linear = InventoryLinearMM(half_spread=0.4, lambda_q=0.0, Q_max=10)

    bid_sym, ask_sym = symmetric.quotes(100.0, 5, 0.25)
    bid_lin, ask_lin = inventory_linear.quotes(100.0, 5, 0.25)

    assert bid_lin == pytest.approx(bid_sym)
    assert ask_lin == pytest.approx(ask_sym)


def test_inventory_linear_skew_direction() -> None:
    symmetric = SymmetricMM(half_spread=0.4)
    inventory_linear = InventoryLinearMM(half_spread=0.4, lambda_q=0.5, Q_max=10)

    bid_sym, ask_sym = symmetric.quotes(100.0, 5, 0.0)
    bid_lin, ask_lin = inventory_linear.quotes(100.0, 5, 0.0)

    assert ask_lin - bid_lin > 0.0
    assert ask_lin < ask_sym
    assert bid_lin < bid_sym


def test_random_quoter_determinism() -> None:
    quoter_a = RandomQuoter(half_spread_max=0.5, rng_seed=42)
    quoter_b = RandomQuoter(half_spread_max=0.5, rng_seed=42)

    sequence_a = [quoter_a.quotes(100.0, 0, 0.0) for _ in range(5)]
    sequence_b = [quoter_b.quotes(100.0, 0, 0.0) for _ in range(5)]

    assert sequence_a == pytest.approx(sequence_b)


def test_random_quoter_returns_positive_spread() -> None:
    quoter = RandomQuoter(half_spread_max=0.5, rng_seed=7)

    for _ in range(100):
        bid, ask = quoter.quotes(100.0, 0, 0.0)
        assert ask >= bid


def test_avs_optimal_wrapper_matches_closed_form() -> None:
    quoter = AvSOptimalMM(sigma=1.2, gamma=0.1, kappa=1.5, T=1.0)

    bid, ask = quoter.quotes(100.0, 3, 0.2)
    reservation = reservation_price(S=100.0, q=3, t=0.2, T=1.0, gamma=0.1, sigma=1.2)
    half_spread = optimal_spread(t=0.2, T=1.0, gamma=0.1, sigma=1.2, kappa=1.5)

    assert bid == pytest.approx(float(reservation) - half_spread)
    assert ask == pytest.approx(float(reservation) + half_spread)
