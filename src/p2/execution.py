"""Shared execution engine for synthetic market-making experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from p2.adverse_selection import ArrivalModel
from p2.config import AdverseSelectionConfig, InventoryConfig, ModelConfig, QueueModelConfig
from p2.hjb_solver import optimal_spread, reservation_price
from p2.inventory import hard_limit_active
from p2.metrics import average_abs_inventory, fill_rate, inventory_variance, pnl_sharpe, spread_capture


class QuoteStrategy(Protocol):
    name: str

    def quotes(
        self,
        mid_prices: np.ndarray,
        inventory: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(slots=True)
class SimResult:
    strategy: str
    pnl_paths: np.ndarray
    inventory_paths: np.ndarray
    fill_count_bid: np.ndarray
    fill_count_ask: np.ndarray
    mean_pnl: float
    std_pnl: float
    sharpe: float
    cash_paths: np.ndarray
    mid_price_paths: np.ndarray
    bid_quote_paths: np.ndarray
    ask_quote_paths: np.ndarray
    bid_fill_paths: np.ndarray
    ask_fill_paths: np.ndarray
    avg_abs_inventory: float
    inventory_variance: float
    spread_capture: float
    avg_bid_fill_rate: float
    avg_ask_fill_rate: float

    @property
    def terminal_inventory(self) -> np.ndarray:
        return self.inventory_paths[:, -1]


@dataclass(slots=True)
class AvellanedaStoikovStrategy:
    model: ModelConfig

    @property
    def name(self) -> str:
        return "avs"

    def quotes(
        self,
        mid_prices: np.ndarray,
        inventory: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        half_spread = optimal_spread(
            t=t,
            T=self.model.T,
            gamma=self.model.gamma,
            sigma=self.model.sigma,
            kappa=self.model.kappa,
        )
        reservation = reservation_price(
            S=mid_prices,
            q=inventory,
            t=t,
            T=self.model.T,
            gamma=self.model.gamma,
            sigma=self.model.sigma,
        )
        reservation_array = np.asarray(reservation, dtype=float)
        return reservation_array - half_spread, reservation_array + half_spread


def simulate_strategy(
    strategy: QuoteStrategy,
    *,
    model: ModelConfig,
    inventory_cfg: InventoryConfig,
    adverse_selection_cfg: AdverseSelectionConfig,
    n_paths: int,
    seed: int,
    use_queue_position: bool = False,
    queue_cfg: QueueModelConfig | None = None,
) -> SimResult:
    if not use_queue_position:
        return _simulate_strategy_instantaneous(
            strategy,
            model=model,
            inventory_cfg=inventory_cfg,
            adverse_selection_cfg=adverse_selection_cfg,
            n_paths=n_paths,
            seed=seed,
        )

    return _simulate_strategy_queue_aware(
        strategy,
        model=model,
        inventory_cfg=inventory_cfg,
        adverse_selection_cfg=adverse_selection_cfg,
        n_paths=n_paths,
        seed=seed,
        queue_cfg=queue_cfg or QueueModelConfig(),
    )


def _simulate_strategy_instantaneous(
    strategy: QuoteStrategy,
    *,
    model: ModelConfig,
    inventory_cfg: InventoryConfig,
    adverse_selection_cfg: AdverseSelectionConfig,
    n_paths: int,
    seed: int,
) -> SimResult:
    n_steps = max(int(round(model.T / model.dt)), 1)
    rng = np.random.default_rng(seed)
    inventory = np.zeros((n_paths, n_steps + 1), dtype=np.int64)
    mid_prices = np.empty((n_paths, n_steps + 1), dtype=float)
    mid_prices[:, 0] = model.initial_mid
    cash = np.zeros(n_paths, dtype=float)
    fill_bid = np.zeros((n_paths, n_steps), dtype=bool)
    fill_ask = np.zeros((n_paths, n_steps), dtype=bool)
    bid_quotes = np.empty((n_paths, n_steps), dtype=float)
    ask_quotes = np.empty((n_paths, n_steps), dtype=float)
    fill_count_bid = np.zeros(n_paths, dtype=np.int64)
    fill_count_ask = np.zeros(n_paths, dtype=np.int64)
    sqrt_dt = float(np.sqrt(model.dt))

    for step in range(n_steps):
        t = step * model.dt
        mid_t = mid_prices[:, step]
        inventory_t = inventory[:, step]
        bid_t, ask_t = strategy.quotes(mid_t, inventory_t, t)
        bid_t = np.asarray(bid_t, dtype=float)
        ask_t = np.asarray(ask_t, dtype=float)
        bid_quotes[:, step] = bid_t
        ask_quotes[:, step] = ask_t

        bid_blocked = np.asarray(hard_limit_active(inventory_t, inventory_cfg.Q_max, direction=1), dtype=bool)
        ask_blocked = np.asarray(hard_limit_active(inventory_t, inventory_cfg.Q_max, direction=-1), dtype=bool)

        delta_bid = np.where(bid_blocked, np.inf, np.maximum(mid_t - bid_t, 0.0))
        delta_ask = np.where(ask_blocked, np.inf, np.maximum(ask_t - mid_t, 0.0))

        lambda_bid = np.asarray(ArrivalModel.intensity(delta_bid, A=model.A, kappa=model.kappa), dtype=float)
        lambda_ask = np.asarray(ArrivalModel.intensity(delta_ask, A=model.A, kappa=model.kappa), dtype=float)
        bid_prob = np.where(bid_blocked, 0.0, 1.0 - np.exp(-lambda_bid * model.dt))
        ask_prob = np.where(ask_blocked, 0.0, 1.0 - np.exp(-lambda_ask * model.dt))

        bid_fill_t = rng.random(n_paths) < bid_prob
        ask_fill_t = rng.random(n_paths) < ask_prob
        fill_bid[:, step] = bid_fill_t
        fill_ask[:, step] = ask_fill_t

        cash -= bid_fill_t * bid_t
        cash += ask_fill_t * ask_t
        fill_count_bid += bid_fill_t.astype(np.int64)
        fill_count_ask += ask_fill_t.astype(np.int64)
        inventory[:, step + 1] = inventory_t + bid_fill_t.astype(np.int64) - ask_fill_t.astype(np.int64)

        adverse_shift = 0.0
        if adverse_selection_cfg.enabled and adverse_selection_cfg.epsilon > 0.0:
            adverse_shift = (
                ask_fill_t.astype(float) * adverse_selection_cfg.epsilon
                - bid_fill_t.astype(float) * adverse_selection_cfg.epsilon
            )

        diffusion = model.sigma * sqrt_dt * rng.standard_normal(n_paths)
        mid_prices[:, step + 1] = mid_t + diffusion + adverse_shift

    pnl = cash + inventory[:, -1] * mid_prices[:, -1]
    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl, ddof=1)) if pnl.size > 1 else 0.0
    return SimResult(
        strategy=strategy.name,
        pnl_paths=pnl,
        inventory_paths=inventory,
        fill_count_bid=fill_count_bid,
        fill_count_ask=fill_count_ask,
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        sharpe=pnl_sharpe(pnl),
        cash_paths=cash,
        mid_price_paths=mid_prices,
        bid_quote_paths=bid_quotes,
        ask_quote_paths=ask_quotes,
        bid_fill_paths=fill_bid,
        ask_fill_paths=fill_ask,
        avg_abs_inventory=average_abs_inventory(inventory),
        inventory_variance=inventory_variance(inventory),
        spread_capture=spread_capture(bid_quotes, ask_quotes, fill_bid, fill_ask),
        avg_bid_fill_rate=fill_rate(fill_count_bid, n_steps, model.dt),
        avg_ask_fill_rate=fill_rate(fill_count_ask, n_steps, model.dt),
    )


def _simulate_strategy_queue_aware(
    strategy: QuoteStrategy,
    *,
    model: ModelConfig,
    inventory_cfg: InventoryConfig,
    adverse_selection_cfg: AdverseSelectionConfig,
    n_paths: int,
    seed: int,
    queue_cfg: QueueModelConfig,
) -> SimResult:
    n_steps = max(int(round(model.T / model.dt)), 1)
    rng = np.random.default_rng(seed)
    inventory = np.zeros((n_paths, n_steps + 1), dtype=np.int64)
    mid_prices = np.empty((n_paths, n_steps + 1), dtype=float)
    mid_prices[:, 0] = model.initial_mid
    cash = np.zeros(n_paths, dtype=float)
    fill_bid = np.zeros((n_paths, n_steps), dtype=bool)
    fill_ask = np.zeros((n_paths, n_steps), dtype=bool)
    bid_quotes = np.empty((n_paths, n_steps), dtype=float)
    ask_quotes = np.empty((n_paths, n_steps), dtype=float)
    fill_count_bid = np.zeros(n_paths, dtype=np.int64)
    fill_count_ask = np.zeros(n_paths, dtype=np.int64)
    sqrt_dt = float(np.sqrt(model.dt))

    quote_tick = 0.01
    bid_queue_depth = np.ones(n_paths, dtype=np.int64)
    ask_queue_depth = np.ones(n_paths, dtype=np.int64)
    bid_own_position = np.full(n_paths, -1, dtype=np.int64)
    ask_own_position = np.full(n_paths, -1, dtype=np.int64)
    active_bid_levels = np.full(n_paths, np.nan, dtype=float)
    active_ask_levels = np.full(n_paths, np.nan, dtype=float)

    for step in range(n_steps):
        t = step * model.dt
        mid_t = mid_prices[:, step]
        inventory_t = inventory[:, step]
        bid_t, ask_t = strategy.quotes(mid_t, inventory_t, t)
        bid_t = np.asarray(bid_t, dtype=float)
        ask_t = np.asarray(ask_t, dtype=float)
        bid_quotes[:, step] = bid_t
        ask_quotes[:, step] = ask_t

        bid_blocked = np.asarray(hard_limit_active(inventory_t, inventory_cfg.Q_max, direction=1), dtype=bool)
        ask_blocked = np.asarray(hard_limit_active(inventory_t, inventory_cfg.Q_max, direction=-1), dtype=bool)

        delta_bid = np.where(bid_blocked, np.inf, np.maximum(mid_t - bid_t, 0.0))
        delta_ask = np.where(ask_blocked, np.inf, np.maximum(ask_t - mid_t, 0.0))
        lambda_bid = np.asarray(ArrivalModel.intensity(delta_bid, A=model.A, kappa=model.kappa), dtype=float)
        lambda_ask = np.asarray(ArrivalModel.intensity(delta_ask, A=model.A, kappa=model.kappa), dtype=float)

        bid_levels = _round_quote_levels(bid_t, quote_tick)
        ask_levels = _round_quote_levels(ask_t, quote_tick)
        bid_replaced = bid_blocked | ~np.isclose(bid_levels, active_bid_levels, equal_nan=True)
        ask_replaced = ask_blocked | ~np.isclose(ask_levels, active_ask_levels, equal_nan=True)

        bid_own_position[bid_replaced & ~bid_blocked] = bid_queue_depth[bid_replaced & ~bid_blocked]
        ask_own_position[ask_replaced & ~ask_blocked] = ask_queue_depth[ask_replaced & ~ask_blocked]
        bid_own_position[bid_blocked] = -1
        ask_own_position[ask_blocked] = -1
        active_bid_levels[bid_blocked] = np.nan
        active_ask_levels[ask_blocked] = np.nan
        active_bid_levels[~bid_blocked] = bid_levels[~bid_blocked]
        active_ask_levels[~ask_blocked] = ask_levels[~ask_blocked]

        bid_fill_t, bid_queue_depth, bid_own_position = _step_queue_side(
            rng=rng,
            queue_depth=bid_queue_depth,
            own_position=bid_own_position,
            lambda_base=lambda_bid,
            dt=model.dt,
            activity_base=model.A,
            arrival_depth_coeff=queue_cfg.arrival_rate_depth_coeff,
            cancel_depth_coeff=queue_cfg.cancel_rate_depth_coeff,
            fill_depth_coeff=queue_cfg.fill_intensity_depth_coeff,
        )
        ask_fill_t, ask_queue_depth, ask_own_position = _step_queue_side(
            rng=rng,
            queue_depth=ask_queue_depth,
            own_position=ask_own_position,
            lambda_base=lambda_ask,
            dt=model.dt,
            activity_base=model.A,
            arrival_depth_coeff=queue_cfg.arrival_rate_depth_coeff,
            cancel_depth_coeff=queue_cfg.cancel_rate_depth_coeff,
            fill_depth_coeff=queue_cfg.fill_intensity_depth_coeff,
        )

        fill_bid[:, step] = bid_fill_t
        fill_ask[:, step] = ask_fill_t

        cash -= bid_fill_t * bid_t
        cash += ask_fill_t * ask_t
        fill_count_bid += bid_fill_t.astype(np.int64)
        fill_count_ask += ask_fill_t.astype(np.int64)
        inventory[:, step + 1] = inventory_t + bid_fill_t.astype(np.int64) - ask_fill_t.astype(np.int64)

        adverse_shift = 0.0
        if adverse_selection_cfg.enabled and adverse_selection_cfg.epsilon > 0.0:
            adverse_shift = (
                ask_fill_t.astype(float) * adverse_selection_cfg.epsilon
                - bid_fill_t.astype(float) * adverse_selection_cfg.epsilon
            )

        diffusion = model.sigma * sqrt_dt * rng.standard_normal(n_paths)
        mid_prices[:, step + 1] = mid_t + diffusion + adverse_shift

    pnl = cash + inventory[:, -1] * mid_prices[:, -1]
    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl, ddof=1)) if pnl.size > 1 else 0.0
    return SimResult(
        strategy=strategy.name,
        pnl_paths=pnl,
        inventory_paths=inventory,
        fill_count_bid=fill_count_bid,
        fill_count_ask=fill_count_ask,
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        sharpe=pnl_sharpe(pnl),
        cash_paths=cash,
        mid_price_paths=mid_prices,
        bid_quote_paths=bid_quotes,
        ask_quote_paths=ask_quotes,
        bid_fill_paths=fill_bid,
        ask_fill_paths=fill_ask,
        avg_abs_inventory=average_abs_inventory(inventory),
        inventory_variance=inventory_variance(inventory),
        spread_capture=spread_capture(bid_quotes, ask_quotes, fill_bid, fill_ask),
        avg_bid_fill_rate=fill_rate(fill_count_bid, n_steps, model.dt),
        avg_ask_fill_rate=fill_rate(fill_count_ask, n_steps, model.dt),
    )


def _round_quote_levels(quotes: np.ndarray, tick_size: float) -> np.ndarray:
    return np.round(np.asarray(quotes, dtype=float) / tick_size) * tick_size


def _step_queue_side(
    *,
    rng: np.random.Generator,
    queue_depth: np.ndarray,
    own_position: np.ndarray,
    lambda_base: np.ndarray,
    dt: float,
    activity_base: float,
    arrival_depth_coeff: float,
    cancel_depth_coeff: float,
    fill_depth_coeff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.maximum(np.asarray(queue_depth, dtype=np.int64), 0)
    position = np.asarray(own_position, dtype=np.int64).copy()
    lambda_arr = np.maximum(np.asarray(lambda_base, dtype=float), 0.0)
    active = position >= 0

    arrival_rate = lambda_arr / (1.0 + arrival_depth_coeff * depth)
    cancel_rate = activity_base * cancel_depth_coeff * np.maximum(depth, 1)
    add_rate = activity_base * fill_depth_coeff * (np.maximum(depth, 1) + 1.0)

    arrivals = rng.poisson(arrival_rate * dt).astype(np.int64)
    cancels = np.minimum(rng.poisson(cancel_rate * dt).astype(np.int64), depth)
    additions = rng.poisson(add_rate * dt).astype(np.int64)

    cancel_ahead = np.zeros_like(depth)
    active_indices = np.flatnonzero(active & (depth > 0) & (cancels > 0) & (position > 0))
    for idx in active_indices:
        cancel_ahead[idx] = int(
            rng.hypergeometric(
                ngood=int(position[idx]),
                nbad=int(max(depth[idx] - position[idx], 0)),
                nsample=int(cancels[idx]),
            )
        )

    position_after_cancel = np.maximum(position - cancel_ahead, 0)
    fills = active & (arrivals > position_after_cancel)
    new_depth = np.maximum(depth - cancels - np.minimum(arrivals, depth), 0) + additions

    position_after = np.maximum(position_after_cancel - arrivals, 0)
    position_after = np.minimum(position_after, new_depth)
    position_after[~active | fills] = -1
    return fills, new_depth.astype(np.int64), position_after.astype(np.int64)
