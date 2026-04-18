import numpy as np
import pytest

from p2.glosten_milgrom import GlostenMilgromLayer, gm_adjusted_intensity


def test_posterior_converges_to_one_on_all_buy_flow() -> None:
    model = GlostenMilgromLayer(mu=0.4, v_high=101.0, v_low=99.0, prior_high=0.5)
    posteriors = model.update_posterior_sequence(np.ones(100, dtype=int))
    assert posteriors[-1] > 0.99


def test_posterior_symmetry_balanced_flow() -> None:
    model = GlostenMilgromLayer(mu=0.4, v_high=101.0, v_low=99.0, prior_high=0.5)
    order_signs = np.tile(np.array([1, -1], dtype=int), 50)
    posteriors = model.update_posterior_sequence(order_signs)
    assert 0.49 <= posteriors[-1] <= 0.51


def test_bid_ask_spread_strictly_positive() -> None:
    value_range = 4.0
    for mu in (0.1, 0.3, 0.5, 0.7):
        model = GlostenMilgromLayer(mu=mu, v_high=102.0, v_low=98.0, prior_high=0.5)
        for prior in (0.3, 0.5, 0.7):
            bid, ask = model.bid_ask_prices(prior=prior)
            assert ask - bid > 0.0
            assert ask - bid <= mu * value_range + 1e-12


def test_intensity_collapses_to_avs_at_prior_half() -> None:
    baseline = 1.5 * np.exp(-2.0 * 0.1)
    lambda_buy, lambda_sell = gm_adjusted_intensity(
        delta=0.1,
        A=1.5,
        kappa=2.0,
        posterior=0.5,
        mu=0.3,
    )
    assert lambda_buy == pytest.approx(baseline, abs=1e-12)
    assert lambda_sell == pytest.approx(baseline, abs=1e-12)


def test_input_validation() -> None:
    invalid_args = (
        {"mu": 0.0, "v_high": 101.0, "v_low": 99.0, "prior_high": 0.5},
        {"mu": 1.0, "v_high": 101.0, "v_low": 99.0, "prior_high": 0.5},
        {"mu": 1.5, "v_high": 101.0, "v_low": 99.0, "prior_high": 0.5},
        {"mu": -0.1, "v_high": 101.0, "v_low": 99.0, "prior_high": 0.5},
        {"mu": 0.3, "v_high": 100.0, "v_low": 100.0, "prior_high": 0.5},
        {"mu": 0.3, "v_high": 99.0, "v_low": 100.0, "prior_high": 0.5},
        {"mu": 0.3, "v_high": 101.0, "v_low": 99.0, "prior_high": 0.0},
        {"mu": 0.3, "v_high": 101.0, "v_low": 99.0, "prior_high": 1.0},
    )

    for kwargs in invalid_args:
        with pytest.raises(ValueError):
            GlostenMilgromLayer(**kwargs)


def test_vectorized_matches_iterative() -> None:
    np.random.seed(0)
    model = GlostenMilgromLayer(mu=0.35, v_high=101.0, v_low=99.0, prior_high=0.4)
    order_signs = np.random.choice(np.array([-1, 1], dtype=int), size=50)

    iterative = np.empty(order_signs.size, dtype=float)
    prior = model.prior_high
    for index, sign in enumerate(order_signs):
        prior = model.posterior(int(sign), prior=prior)
        iterative[index] = prior

    vectorized = model.update_posterior_sequence(order_signs)
    np.testing.assert_allclose(vectorized, iterative, atol=1e-12, rtol=0.0)
