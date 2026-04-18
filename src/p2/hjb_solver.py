"""Closed-form components of the Avellaneda-Stoikov market-making model."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _to_output(value: np.ndarray | float) -> float | np.ndarray:
    if np.isscalar(value):
        return float(value)
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array.item())
    return array


def reservation_price(
    S: float | np.ndarray,
    q: float | np.ndarray,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> float | np.ndarray:
    """Avellaneda-Stoikov reservation price from Eq. 7."""

    tau = max(T - t, 0.0)
    price = np.asarray(S, dtype=float) - np.asarray(q, dtype=float) * gamma * sigma**2 * tau
    return _to_output(price)


def optimal_total_spread(t: float, T: float, gamma: float, sigma: float, kappa: float) -> float:
    """Total optimal spread from Avellaneda-Stoikov Eq. 8/9."""

    tau = max(T - t, 0.0)
    if math.isclose(gamma, 0.0, abs_tol=1e-12):
        return 2.0 / kappa
    return gamma * sigma**2 * tau + (2.0 / gamma) * math.log1p(gamma / kappa)


def optimal_spread(t: float, T: float, gamma: float, sigma: float, kappa: float) -> float:
    """One-sided quote distance used for bid/ask placement."""

    return 0.5 * optimal_total_spread(t=t, T=T, gamma=gamma, sigma=sigma, kappa=kappa)


def optimal_bid(S: float, q: float, t: float, T: float, gamma: float, sigma: float, kappa: float) -> float:
    """Optimal bid quote using the reservation price and half-spread."""

    r_t = reservation_price(S=S, q=q, t=t, T=T, gamma=gamma, sigma=sigma)
    half_spread = optimal_spread(t=t, T=T, gamma=gamma, sigma=sigma, kappa=kappa)
    return float(r_t - half_spread)


def optimal_ask(S: float, q: float, t: float, T: float, gamma: float, sigma: float, kappa: float) -> float:
    """Optimal ask quote using the reservation price and half-spread."""

    r_t = reservation_price(S=S, q=q, t=t, T=T, gamma=gamma, sigma=sigma)
    half_spread = optimal_spread(t=t, T=T, gamma=gamma, sigma=sigma, kappa=kappa)
    return float(r_t + half_spread)
