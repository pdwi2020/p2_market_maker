"""Baseline market-making strategies and strategy comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from p2.config import AdverseSelectionConfig, InventoryConfig, ModelConfig
from p2.execution import SimResult, simulate_strategy
from p2.hjb_solver import optimal_spread


@dataclass(slots=True)
class SymmetricMM:
    half_spread: float
    name: str = "symmetric"

    def quotes(
        self,
        S: float | np.ndarray,
        q: float | np.ndarray,
        t: float,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        s_arr = np.asarray(S, dtype=float)
        bid = s_arr - self.half_spread
        ask = s_arr + self.half_spread
        if bid.ndim == 0:
            return float(bid.item()), float(ask.item())
        return bid, ask


@dataclass(slots=True)
class SymmetricAS:
    sigma: float
    gamma: float
    kappa: float
    T: float
    name: str = "symmetric"

    def quotes(
        self,
        S: float | np.ndarray,
        q: float | np.ndarray,
        t: float,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        del q
        s_arr = np.asarray(S, dtype=float)
        half_spread = optimal_spread(t=t, T=self.T, gamma=self.gamma, sigma=self.sigma, kappa=self.kappa)
        bid = s_arr - half_spread
        ask = s_arr + half_spread
        if bid.ndim == 0:
            return float(bid.item()), float(ask.item())
        return bid, ask


@dataclass(slots=True)
class ConstantSpreadMM:
    sigma: float
    gamma: float
    kappa: float
    T: float
    half_spread: float = field(init=False)
    name: str = "constant_spread"

    def __post_init__(self) -> None:
        self.half_spread = optimal_spread(t=0.0, T=self.T, gamma=self.gamma, sigma=self.sigma, kappa=self.kappa)

    def quotes(
        self,
        S: float | np.ndarray,
        q: float | np.ndarray,
        t: float,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        s_arr = np.asarray(S, dtype=float)
        bid = s_arr - self.half_spread
        ask = s_arr + self.half_spread
        if bid.ndim == 0:
            return float(bid.item()), float(ask.item())
        return bid, ask


def _simulate_quotes(
    strategy: Any,
    n_paths: int,
    *,
    sigma: float = 1.0,
    gamma: float = 0.1,
    kappa: float = 1.5,
    A: float = 140.0,
    T: float = 1.0,
    dt: float = 0.001,
    Q_max: int = 10,
    seed: int = 42,
    s0: float = 100.0,
    adverse_selection: AdverseSelectionConfig | None = None,
) -> SimResult:
    model = ModelConfig(sigma=sigma, gamma=gamma, kappa=kappa, A=A, T=T, dt=dt, initial_mid=s0)
    inventory_cfg = InventoryConfig(Q_max=Q_max)
    adverse_cfg = adverse_selection or AdverseSelectionConfig(enabled=True, epsilon=0.02)
    return simulate_strategy(
        strategy,
        model=model,
        inventory_cfg=inventory_cfg,
        adverse_selection_cfg=adverse_cfg,
        n_paths=n_paths,
        seed=seed,
    )


def compare_strategies(simulators: dict[str, Any], n_paths: int) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for name, spec in simulators.items():
        runner = spec
        sim_kwargs: dict[str, Any] = {}

        if isinstance(spec, tuple) and len(spec) == 2:
            runner, sim_kwargs = spec

        if hasattr(runner, "run"):
            result = runner.run(n_paths)
        elif hasattr(runner, "quotes"):
            result = _simulate_quotes(runner, n_paths=n_paths, **sim_kwargs)
        elif callable(runner):
            result = runner(n_paths)
        else:
            raise TypeError(f"Unsupported strategy runner for '{name}': {type(runner)!r}")

        rows.append(
            {
                "strategy": name,
                "mean_pnl": float(result.mean_pnl),
                "std_pnl": float(result.std_pnl),
                "sharpe": float(result.sharpe),
                "avg_abs_inventory": float(result.avg_abs_inventory),
                "inventory_variance": float(result.inventory_variance),
                "spread_capture": float(result.spread_capture),
                "avg_bid_fill_rate": float(result.avg_bid_fill_rate),
                "avg_ask_fill_rate": float(result.avg_ask_fill_rate),
            }
        )

    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


@dataclass(slots=True)
class InventoryLinearMM:
    half_spread: float
    lambda_q: float = 1.0
    Q_max: int = 10
    name: str = "inventory_linear"

    def quotes(
        self,
        S: float | np.ndarray,
        q: float | np.ndarray,
        t: float,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        del t
        s_arr = np.asarray(S, dtype=float)
        q_arr = np.asarray(q, dtype=float)
        skew = self.lambda_q * q_arr / max(self.Q_max, 1)
        bid = s_arr - self.half_spread * (1.0 + skew)
        ask = s_arr + self.half_spread * (1.0 - skew)
        if bid.ndim == 0:
            return float(bid.item()), float(ask.item())
        return bid, ask


@dataclass(slots=True)
class RandomQuoter:
    half_spread_max: float
    rng_seed: int = 0
    name: str = "random"
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def quotes(
        self,
        S: float | np.ndarray,
        q: float | np.ndarray,
        t: float,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        del q, t
        s_arr = np.asarray(S, dtype=float)
        bid_delta = self._rng.uniform(0.0, 2.0 * self.half_spread_max, size=s_arr.shape)
        ask_delta = self._rng.uniform(0.0, 2.0 * self.half_spread_max, size=s_arr.shape)
        bid = s_arr - bid_delta
        ask = s_arr + ask_delta
        if bid.ndim == 0:
            return float(bid.item()), float(ask.item())
        return bid, ask


@dataclass(slots=True)
class AvSOptimalMM:
    sigma: float
    gamma: float
    kappa: float
    T: float
    name: str = "avs_optimal"

    def quotes(
        self,
        S: float | np.ndarray,
        q: float | np.ndarray,
        t: float,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        from p2.hjb_solver import reservation_price

        half_spread = optimal_spread(t=t, T=self.T, gamma=self.gamma, sigma=self.sigma, kappa=self.kappa)
        reservation = np.asarray(
            reservation_price(S=S, q=q, t=t, T=self.T, gamma=self.gamma, sigma=self.sigma),
            dtype=float,
        )
        bid = reservation - half_spread
        ask = reservation + half_spread
        if bid.ndim == 0:
            return float(bid.item()), float(ask.item())
        return bid, ask
