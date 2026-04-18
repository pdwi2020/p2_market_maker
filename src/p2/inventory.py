"""Inventory limit and quote-skew helpers."""

from __future__ import annotations

import numpy as np


def hard_limit_active(q: int | np.ndarray, Q_max: int, direction: int) -> bool | np.ndarray:
    q_arr = np.asarray(q)
    if direction > 0:
        blocked = q_arr >= Q_max
    elif direction < 0:
        blocked = q_arr <= -Q_max
    else:
        blocked = np.zeros_like(q_arr, dtype=bool)
    if blocked.ndim == 0:
        return bool(blocked.item())
    return blocked


def adjusted_quotes(
    bid: float | np.ndarray,
    ask: float | np.ndarray,
    q: int | np.ndarray,
    Q_max: int,
    skew_factor: float = 0.5,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    bid_arr = np.asarray(bid, dtype=float)
    ask_arr = np.asarray(ask, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    half_spread = 0.5 * (ask_arr - bid_arr)
    mid = 0.5 * (ask_arr + bid_arr)
    inventory_ratio = np.clip(q_arr / max(Q_max, 1), -1.0, 1.0)
    shift = skew_factor * inventory_ratio * half_spread
    adj_bid = mid - shift - half_spread
    adj_ask = mid - shift + half_spread
    if adj_bid.ndim == 0:
        return float(adj_bid.item()), float(adj_ask.item())
    return adj_bid, adj_ask
