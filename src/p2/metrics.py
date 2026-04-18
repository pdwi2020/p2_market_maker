"""Utility metrics for simulation and backtesting outputs."""

from __future__ import annotations

import numpy as np


def terminal_pnl_mean(pnl: np.ndarray) -> float:
    pnl_arr = np.asarray(pnl, dtype=float)
    return float(np.mean(pnl_arr)) if pnl_arr.size else 0.0


def terminal_pnl_std(pnl: np.ndarray) -> float:
    pnl_arr = np.asarray(pnl, dtype=float)
    return float(np.std(pnl_arr, ddof=1)) if pnl_arr.size > 1 else 0.0


def pnl_sharpe(pnl: np.ndarray) -> float:
    std = terminal_pnl_std(pnl)
    mean = terminal_pnl_mean(pnl)
    return mean / std if std > 0 else 0.0


def inventory_variance(inventory: np.ndarray) -> float:
    inventory_arr = np.asarray(inventory, dtype=float)
    return float(np.var(inventory_arr))


def average_abs_inventory(inventory: np.ndarray) -> float:
    inventory_arr = np.asarray(inventory, dtype=float)
    return float(np.mean(np.abs(inventory_arr))) if inventory_arr.size else 0.0


def fill_rate(fill_counts: np.ndarray, n_steps: int, dt: float) -> float:
    fills = np.asarray(fill_counts, dtype=float)
    horizon = max(n_steps * dt * max(fills.size, 1), 1e-12)
    return float(np.sum(fills) / horizon)


def spread_capture(
    bid_quotes: np.ndarray,
    ask_quotes: np.ndarray,
    fill_bid: np.ndarray,
    fill_ask: np.ndarray,
) -> float:
    bid_arr = np.asarray(bid_quotes, dtype=float)
    ask_arr = np.asarray(ask_quotes, dtype=float)
    fill_bid_arr = np.asarray(fill_bid, dtype=float)
    fill_ask_arr = np.asarray(fill_ask, dtype=float)
    spread = ask_arr - bid_arr
    realized = 0.5 * spread * (fill_bid_arr + fill_ask_arr)
    return float(np.mean(np.sum(realized, axis=1))) if realized.ndim == 2 and realized.size else 0.0
