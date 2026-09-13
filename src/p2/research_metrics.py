"""Daily performance statistics for the preregistered study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PerformanceSummary:
    """Aggregate statistics for a daily net-PnL series."""

    days: int
    total_net_pnl: float
    mean_daily_pnl: float
    annualized_sharpe: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    max_drawdown: float
    deflated_sharpe_ratio: float


def annualized_sharpe(values: Sequence[float], *, periods_per_year: int = 365) -> float:
    """Compute annualized Sharpe with zero risk-free rate."""
    samples = np.asarray(values, dtype=np.float64)
    if len(samples) < 2:
        return np.nan
    scale = float(np.std(samples, ddof=1))
    if scale == 0.0 or not np.isfinite(scale):
        return np.nan
    return float(np.sqrt(periods_per_year) * np.mean(samples) / scale)


def max_drawdown(values: Sequence[float]) -> float:
    """Return absolute peak-to-trough drawdown of cumulative PnL."""
    samples = np.asarray(values, dtype=np.float64)
    curve = np.r_[0.0, np.cumsum(samples)]
    peaks = np.maximum.accumulate(curve)
    return float(np.max(peaks - curve))


def moving_block_bootstrap_sharpe(
    values: Sequence[float],
    *,
    block_size: int = 7,
    resamples: int = 10_000,
    seed: int = 20_250_701,
    periods_per_year: int = 365,
) -> tuple[float, float]:
    """Return a circular moving-block 95% interval for annualized Sharpe."""
    samples = np.asarray(values, dtype=np.float64)
    if len(samples) < 2 or block_size <= 0 or resamples <= 0:
        raise ValueError("bootstrap requires data, a positive block, and positive resamples")
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(len(samples) / block_size)
    offsets = np.arange(block_size)
    sharpes = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        starts = rng.integers(0, len(samples), size=blocks_needed)
        positions = (starts[:, None] + offsets[None, :]) % len(samples)
        bootstrap_sample = samples[positions.reshape(-1)[: len(samples)]]
        sharpes[index] = annualized_sharpe(
            bootstrap_sample, periods_per_year=periods_per_year
        )
    finite = sharpes[np.isfinite(sharpes)]
    if len(finite) == 0:
        return np.nan, np.nan
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def deflated_sharpe_ratio(
    values: Sequence[float],
    trial_sharpes: Sequence[float],
    *,
    trial_count: int = 31,
) -> float:
    """Compute the selection- and non-normality-adjusted Sharpe probability."""
    samples = np.asarray(values, dtype=np.float64)
    trials = np.asarray(trial_sharpes, dtype=np.float64)
    trials = trials[np.isfinite(trials)]
    if len(samples) < 3 or len(trials) < 2:
        return np.nan
    if trial_count < 2 or len(trials) != trial_count:
        raise ValueError("trial_sharpes must contain the preregistered trial count")
    sample_scale = np.std(samples, ddof=1)
    if sample_scale == 0.0 or not np.isfinite(sample_scale):
        return np.nan
    sharpe = float(np.mean(samples) / sample_scale)
    trial_variance = float(np.var(trials, ddof=1))
    if trial_variance < 0.0 or not np.isfinite(trial_variance):
        return np.nan
    normal = NormalDist()
    euler_gamma = 0.5772156649015329
    expected_max = np.sqrt(trial_variance) * (
        (1.0 - euler_gamma) * normal.inv_cdf(1.0 - 1.0 / trial_count)
        + euler_gamma * normal.inv_cdf(1.0 - 1.0 / (trial_count * np.e))
    )
    centered = samples - np.mean(samples)
    population_scale = np.std(samples)
    if population_scale == 0.0:
        return np.nan
    normalized = centered / population_scale
    skewness = float(np.mean(normalized**3))
    kurtosis = float(np.mean(normalized**4))
    denominator_sq = (
        1.0
        - skewness * sharpe
        + 0.25 * (kurtosis - 1.0) * sharpe**2
    )
    if denominator_sq <= 0.0:
        return np.nan
    statistic = (sharpe - expected_max) * np.sqrt(len(samples) - 1)
    statistic /= np.sqrt(denominator_sq)
    return float(normal.cdf(statistic))


def summarize_performance(
    values: Sequence[float],
    trial_sharpes: Sequence[float],
    *,
    bootstrap_resamples: int = 10_000,
) -> PerformanceSummary:
    """Compute the complete preregistered daily-PnL summary."""
    samples = np.asarray(values, dtype=np.float64)
    low, high = moving_block_bootstrap_sharpe(
        samples, resamples=bootstrap_resamples
    )
    return PerformanceSummary(
        days=len(samples),
        total_net_pnl=float(np.sum(samples)),
        mean_daily_pnl=float(np.mean(samples)),
        annualized_sharpe=annualized_sharpe(samples),
        sharpe_ci_low=low,
        sharpe_ci_high=high,
        max_drawdown=max_drawdown(samples),
        deflated_sharpe_ratio=deflated_sharpe_ratio(samples, trial_sharpes),
    )
