"""Glosten-Milgrom (1985) Bayesian adverse-selection layer.

Models a market where a fraction `mu` of traders are informed about a
latent value V ∈ {v_low, v_high}. Informed traders buy at the ask iff
V = v_high and sell at the bid iff V = v_low. Uninformed traders flip
a fair coin. The market maker observes order flow and updates the
posterior P(V = v_high | order history) by Bayes. This is the
canonical noisy-rational model of adverse selection and extends the
Avellaneda-Stoikov framework with informed-trader-aware intensities.

Reference: Glosten, L.R. & Milgrom, P.R. (1985). "Bid, ask and
transaction prices in a specialist market with heterogeneously
informed traders." Journal of Financial Economics 14, 71-100.
"""

from __future__ import annotations

import math

import numpy as np


class GlostenMilgromLayer:
    """Bayesian adverse-selection layer with informed and noise traders."""

    def __init__(self, mu: float, v_high: float, v_low: float, prior_high: float = 0.5) -> None:
        """Initialize model parameters and validate the prior/value support."""

        if not 0.0 < mu < 1.0:
            raise ValueError("mu must satisfy 0 < mu < 1")
        if not v_high > v_low:
            raise ValueError("v_high must be strictly greater than v_low")
        if not 0.0 < prior_high < 1.0:
            raise ValueError("prior_high must satisfy 0 < prior_high < 1")

        self.mu = float(mu)
        self.v_high = float(v_high)
        self.v_low = float(v_low)
        self.prior_high = float(prior_high)
        self._buy_likelihood_high = 0.5 + 0.5 * self.mu
        self._buy_likelihood_low = 0.5 - 0.5 * self.mu
        self._log_likelihood_ratio = math.log(self._buy_likelihood_high / self._buy_likelihood_low)

    @staticmethod
    def _clamp_prior(prior: float) -> float:
        """Clamp priors to the open unit interval to keep Bayes updates stable."""

        if not 0.0 <= prior <= 1.0:
            raise ValueError("prior must satisfy 0 <= prior <= 1")
        eps = np.finfo(float).eps
        return float(np.clip(prior, eps, 1.0 - eps))

    @staticmethod
    def _validate_order_sign(order_sign: int) -> int:
        """Validate that the order sign is +1 for buy or -1 for sell."""

        sign = int(order_sign)
        if sign not in (-1, 1):
            raise ValueError("order_sign must be +1 for buy or -1 for sell")
        return sign

    @staticmethod
    def _sigmoid(log_odds: np.ndarray | float) -> np.ndarray | float:
        """Convert log-odds to probabilities with a numerically stable sigmoid."""

        values = np.asarray(log_odds, dtype=float)
        probs = np.where(
            values >= 0.0,
            1.0 / (1.0 + np.exp(-values)),
            np.exp(values) / (1.0 + np.exp(values)),
        )
        if probs.ndim == 0:
            return float(probs.item())
        return probs

    def posterior(self, order_sign: int, prior: float | None = None) -> float:
        """Return the posterior probability of the high-value state after one order."""

        sign = self._validate_order_sign(order_sign)
        current_prior = self._clamp_prior(self.prior_high if prior is None else prior)
        log_odds = math.log(current_prior / (1.0 - current_prior)) + sign * self._log_likelihood_ratio
        return float(self._sigmoid(log_odds))

    def expected_value_given_buy(self, prior: float) -> float:
        """Return the expected latent value conditional on observing a buy next."""

        posterior_buy = self.posterior(order_sign=1, prior=prior)
        return float(self.v_high * posterior_buy + self.v_low * (1.0 - posterior_buy))

    def expected_value_given_sell(self, prior: float) -> float:
        """Return the expected latent value conditional on observing a sell next."""

        posterior_sell = self.posterior(order_sign=-1, prior=prior)
        return float(self.v_high * posterior_sell + self.v_low * (1.0 - posterior_sell))

    def bid_ask_prices(self, prior: float) -> tuple[float, float]:
        """Map conditional expectations into Glosten-Milgrom bid and ask prices."""

        bid = self.expected_value_given_sell(prior=prior)
        ask = self.expected_value_given_buy(prior=prior)
        return bid, ask

    def update_posterior_sequence(self, order_signs: np.ndarray, prior: float | None = None) -> np.ndarray:
        """Vectorize posterior updates over a sequence using cumulative log-odds."""

        signs = np.asarray(order_signs, dtype=int)
        if signs.ndim != 1:
            raise ValueError("order_signs must be a one-dimensional array")
        if signs.size == 0:
            return np.empty(0, dtype=float)
        if not np.all(np.isin(signs, (-1, 1))):
            raise ValueError("order_signs must contain only +1 and -1")

        current_prior = self._clamp_prior(self.prior_high if prior is None else prior)
        initial_log_odds = math.log(current_prior / (1.0 - current_prior))
        log_odds_path = initial_log_odds + self._log_likelihood_ratio * np.cumsum(signs, dtype=float)
        return np.asarray(self._sigmoid(log_odds_path), dtype=float)


def gm_adjusted_intensity(
    delta: float,
    A: float,
    kappa: float,
    posterior: float,
    mu: float,
) -> tuple[float, float]:
    """Skew AvS intensities toward the side favored by the informed-trader posterior.

    The baseline Avellaneda-Stoikov intensity is `A * exp(-kappa * delta)`.
    We scale that baseline by a posterior-weighted informed-flow factor:
    `1 - mu + 2 * mu * posterior` for buys and
    `1 - mu + 2 * mu * (1 - posterior)` for sells.
    This preserves the AvS baseline exactly at `posterior = 0.5`.
    """

    if not 0.0 <= posterior <= 1.0:
        raise ValueError("posterior must satisfy 0 <= posterior <= 1")
    if not 0.0 <= mu <= 1.0:
        raise ValueError("mu must satisfy 0 <= mu <= 1")

    baseline = float(A * np.exp(-kappa * delta))
    lambda_buy = baseline * (1.0 - mu + 2.0 * mu * posterior)
    lambda_sell = baseline * (1.0 - mu + 2.0 * mu * (1.0 - posterior))
    return float(lambda_buy), float(lambda_sell)
