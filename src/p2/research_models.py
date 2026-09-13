"""Preregistered quote formulas, calibration, and fill accounting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_DEPTH_GRID = np.arange(0.05, 2.0, 0.1)


@dataclass(frozen=True)
class DailyCalibration:
    """Parameters estimated from one preceding UTC date."""

    A: float
    kappa: float
    sigma: float
    valid_seconds: int
    trade_count: int


@dataclass(frozen=True)
class PnLDecomposition:
    """Additive end-of-day PnL components."""

    realized_spread: float
    inventory_revaluation: float
    fees: float
    net_pnl: float


def symmetric_touch_quotes(best_bid: float, best_ask: float) -> tuple[float, float]:
    """Join the displayed touch."""
    if not np.isfinite(best_bid) or not np.isfinite(best_ask) or best_bid >= best_ask:
        raise ValueError("best bid and ask must form a valid spread")
    return float(best_bid), float(best_ask)


def avellaneda_stoikov_quotes(
    mid: float,
    inventory: float,
    elapsed_seconds: float,
    *,
    gamma: float,
    sigma: float,
    kappa: float,
    horizon_seconds: float = 86_400.0,
) -> tuple[float, float]:
    """Return finite-horizon Avellaneda-Stoikov bid and ask prices."""
    if gamma <= 0.0 or sigma <= 0.0 or kappa <= 0.0 or horizon_seconds <= 0.0:
        raise ValueError("quote parameters must be positive")
    remaining = max(float(horizon_seconds) - float(elapsed_seconds), 0.0)
    reservation = float(mid) - float(inventory) * gamma * sigma**2 * remaining
    half_spread = 0.5 * gamma * sigma**2 * remaining
    half_spread += np.log1p(gamma / kappa) / gamma
    return reservation - half_spread, reservation + half_spread


def glft_quotes(
    mid: float,
    inventory: float,
    *,
    gamma: float,
    sigma: float,
    A: float,
    kappa: float,
    order_size: float = 0.01,
) -> tuple[float, float]:
    """Return finite-order-size asymptotic GLFT bid and ask prices."""
    if min(gamma, sigma, A, kappa, order_size) <= 0.0:
        raise ValueError("quote parameters must be positive")
    scaled_risk = gamma * order_size
    log_base = np.log1p(scaled_risk / kappa)
    c1 = log_base / scaled_risk
    log_power = (kappa / scaled_risk + 1.0) * log_base
    c2 = np.sqrt(
        gamma / (2.0 * A * order_size * kappa) * np.exp(log_power)
    )
    reservation = float(mid) - float(inventory) * sigma * c2
    half_spread = c1 + 0.5 * order_size * sigma * c2
    return reservation - half_spread, reservation + half_spread


def apply_imbalance_skew(
    bid: float,
    ask: float,
    imbalance: float,
    *,
    beta: float,
    tick_size: float,
) -> tuple[float, float]:
    """Shift both quotes in the direction of touch imbalance."""
    if beta < 0.0 or tick_size <= 0.0:
        raise ValueError("beta must be nonnegative and tick_size must be positive")
    if not -1.0 <= imbalance <= 1.0:
        raise ValueError("imbalance must be between -1 and 1")
    shift = beta * tick_size * imbalance
    return float(bid + shift), float(ask + shift)


def fit_log_linear_intensity(
    depths: Sequence[float], intensities: Sequence[float]
) -> tuple[float, float]:
    """Fit log lambda equals log A minus kappa times depth."""
    depth_values = np.asarray(depths, dtype=np.float64)
    intensity_values = np.asarray(intensities, dtype=np.float64)
    keep = np.isfinite(depth_values) & np.isfinite(intensity_values)
    keep &= intensity_values > 0.0
    if np.count_nonzero(keep) < 5:
        raise ValueError("at least five positive intensity points are required")
    slope, intercept = np.polyfit(depth_values[keep], np.log(intensity_values[keep]), 1)
    A = float(np.exp(intercept))
    kappa = float(-slope)
    if not np.isfinite(A) or not np.isfinite(kappa) or A <= 0.0 or kappa <= 0.0:
        raise ValueError("intensity fit must produce positive A and kappa")
    return A, kappa


def calibrate_arrival_intensity(
    trade_depths: Sequence[float],
    observed_seconds: float,
    *,
    depth_grid: Sequence[float] = DEFAULT_DEPTH_GRID,
) -> tuple[float, float]:
    """Estimate exponential arrival intensity from cumulative trade depths."""
    if observed_seconds <= 0.0:
        raise ValueError("observed_seconds must be positive")
    trade_values = np.asarray(trade_depths, dtype=np.float64)
    trade_values = trade_values[np.isfinite(trade_values) & (trade_values >= 0.0)]
    grid = np.asarray(depth_grid, dtype=np.float64)
    intensities = np.count_nonzero(trade_values[:, None] >= grid[None, :], axis=0)
    intensities = intensities.astype(np.float64) / observed_seconds
    return fit_log_linear_intensity(grid, intensities)


def calibrate_one_second_sigma(
    midpoints: Sequence[float],
    seconds: Sequence[int],
    *,
    minimum_changes: int = 3_600,
) -> tuple[float, int]:
    """Estimate price volatility from contiguous one-second midpoint changes."""
    mids = np.asarray(midpoints, dtype=np.float64)
    second_values = np.asarray(seconds, dtype=np.int64)
    if mids.shape != second_values.shape:
        raise ValueError("midpoints and seconds must have matching shapes")
    contiguous = np.diff(second_values) == 1
    changes = np.diff(mids)[contiguous]
    changes = changes[np.isfinite(changes)]
    if len(changes) < minimum_changes:
        raise ValueError(f"at least {minimum_changes} valid midpoint changes are required")
    sigma = float(np.std(changes, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    return sigma, int(len(changes))


def calibrate_bybit_day(cache_path: str | Path) -> DailyCalibration:
    """Estimate daily parameters from a reconstructed Bybit event stream."""
    table = pq.read_table(
        Path(cache_path),
        columns=[
            "time",
            "event_type",
            "book_valid",
            "best_bid",
            "best_ask",
            "trade_price",
            "trade_side",
        ],
    )
    book = table.filter(pc.equal(table["event_type"], "book"))
    book_times = book["time"].to_numpy(zero_copy_only=False)
    book_ms = book_times.astype("datetime64[ms]").astype(np.int64)
    valid = book["book_valid"].to_numpy(zero_copy_only=False).astype(bool)
    bids = book["best_bid"].to_numpy(zero_copy_only=False).astype(np.float64)
    asks = book["best_ask"].to_numpy(zero_copy_only=False).astype(np.float64)
    valid &= np.isfinite(bids) & np.isfinite(asks) & (bids < asks)
    seconds = book_ms // 1_000
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        raise ValueError("no valid book observations are available")
    valid_seconds = seconds[valid_indices]
    boundaries = np.r_[valid_seconds[1:] != valid_seconds[:-1], True]
    sampled_indices = valid_indices[boundaries]
    sampled_seconds = seconds[sampled_indices]
    sampled_midpoints = 0.5 * (bids[sampled_indices] + asks[sampled_indices])
    sigma, observed_changes = calibrate_one_second_sigma(sampled_midpoints, sampled_seconds)

    trades = table.filter(pc.equal(table["event_type"], "trade"))
    trade_times = trades["time"].to_numpy(zero_copy_only=False)
    trade_ms = trade_times.astype("datetime64[ms]").astype(np.int64)
    trade_prices = trades["trade_price"].to_numpy(zero_copy_only=False).astype(np.float64)
    trade_sides = np.asarray(trades["trade_side"].to_pylist(), dtype=object)
    positions = np.searchsorted(book_ms, trade_ms, side="right") - 1
    usable = positions >= 0
    safe_positions = np.maximum(positions, 0)
    usable &= valid[safe_positions]
    trade_prices = trade_prices[usable]
    trade_sides = trade_sides[usable]
    trade_midpoints = 0.5 * (
        bids[safe_positions[usable]] + asks[safe_positions[usable]]
    )
    buy = np.char.lower(trade_sides.astype(str)) == "buy"
    depths = np.where(buy, trade_prices - trade_midpoints, trade_midpoints - trade_prices)
    A, kappa = calibrate_arrival_intensity(depths, float(observed_changes))
    return DailyCalibration(
        A=A,
        kappa=kappa,
        sigma=sigma,
        valid_seconds=observed_changes,
        trade_count=len(trade_prices),
    )


def fill_fee(price: float, size: float, fee_rate: float) -> float:
    """Return the signed fee charged on absolute fill notional."""
    return abs(float(price) * float(size)) * float(fee_rate)


def decompose_fills(
    inventory_changes: Sequence[float],
    fill_prices: Sequence[float],
    fill_midpoints: Sequence[float],
    fee_rates: Sequence[float],
    *,
    final_midpoint: float,
) -> PnLDecomposition:
    """Compute additive spread, inventory, fee, and net PnL totals."""
    changes = np.asarray(inventory_changes, dtype=np.float64)
    prices = np.asarray(fill_prices, dtype=np.float64)
    mids = np.asarray(fill_midpoints, dtype=np.float64)
    rates = np.asarray(fee_rates, dtype=np.float64)
    if not (changes.shape == prices.shape == mids.shape == rates.shape):
        raise ValueError("fill arrays must have matching shapes")
    realized_spread = float(np.sum(-changes * (prices - mids)))
    inventory_revaluation = float(np.sum(changes * (float(final_midpoint) - mids)))
    fees = float(np.sum(np.abs(changes * prices) * rates))
    net_pnl = realized_spread + inventory_revaluation - fees
    return PnLDecomposition(
        realized_spread=realized_spread,
        inventory_revaluation=inventory_revaluation,
        fees=fees,
        net_pnl=net_pnl,
    )
