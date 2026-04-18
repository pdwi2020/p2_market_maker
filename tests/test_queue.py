from dataclasses import dataclass

import numpy as np
import pytest

from p2.config import AdverseSelectionConfig, InventoryConfig, ModelConfig
from p2.execution import simulate_strategy
from p2.queue import (
    QueueReactiveModel,
    QueueState,
    calibrate_queue_reactive,
    expected_time_to_fill,
    queue_reactive_dynamics,
)


@dataclass(slots=True)
class BidOnlyStrategy:
    name: str = "bid_only"

    def quotes(self, mid_prices: np.ndarray, inventory: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        bid = np.asarray(mid_prices, dtype=float)
        ask = np.asarray(mid_prices, dtype=float) + 1e6
        return bid, ask


def test_fifo_filling() -> None:
    initial = QueueState(queue_size=5, own_position=3)
    trajectory = queue_reactive_dynamics(
        [
            {"event": "trade", "size": 2},
            {"event": "trade", "size": 1},
            {"event": "trade", "size": 1},
        ],
        initial,
    )

    assert trajectory[0].own_position == 3
    assert trajectory[1].own_position == 1
    assert trajectory[2].own_position == 0
    assert trajectory[3].own_position is None


def test_cancellation_reduces_depth() -> None:
    initial = QueueState(queue_size=4, own_position=7)
    trajectory = queue_reactive_dynamics(
        [
            {"event_type": 2, "size": 2, "ahead_fraction": 0.5},
            {"event": "own_cancel"},
        ],
        initial,
    )

    assert initial.own_position == 4
    assert trajectory[1].queue_size == 2
    assert trajectory[1].own_position == 2
    assert trajectory[2].own_position is None


def test_fill_probability_monotonic() -> None:
    probabilities = [
        QueueReactiveModel.fill_probability(
            own_position=position,
            queue_depth=12,
            arrival_rate=3.0,
            cancel_rate=1.25,
            dt=0.2,
        )
        for position in range(5)
    ]

    assert all(left > right for left, right in zip(probabilities, probabilities[1:]))
    assert expected_time_to_fill(own_position=4, queue_depth=12, arrival_rate=3.0, cancel_rate=1.25) > expected_time_to_fill(
        own_position=1,
        queue_depth=12,
        arrival_rate=3.0,
        cancel_rate=1.25,
    )


def test_calibration_round_trip() -> None:
    rng = np.random.default_rng(1234)
    depth_grid = [1, 5, 10]
    true_arrival = {1: 1.6, 5: 1.1, 10: 0.7}
    true_cancel = {1: 0.3, 5: 0.6, 10: 1.0}
    true_add = {1: 0.8, 5: 1.0, 10: 1.4}
    events: list[dict[str, float]] = []

    dt = 0.05
    for depth in depth_grid:
        for _ in range(8_000):
            events.append(
                {
                    "queue_depth": float(depth),
                    "dt": dt,
                    "arrival_count": float(rng.poisson(true_arrival[depth] * dt)),
                    "cancel_count": float(rng.poisson(true_cancel[depth] * dt)),
                    "add_count": float(rng.poisson(true_add[depth] * dt)),
                }
            )

    model = calibrate_queue_reactive(events, depth_grid)
    assert model.depth_grid is not None
    assert model.arrival_rates is not None
    assert model.cancel_rates is not None
    assert model.fill_intensities is not None

    for depth in depth_grid:
        assert model.arrival_rate_fn(depth) == pytest.approx(true_arrival[depth], rel=0.15)
        assert model.cancel_rate_fn(depth) == pytest.approx(true_cancel[depth], rel=0.2)
        assert model.fill_intensity_fn(depth) == pytest.approx(true_add[depth], rel=0.15)

    model.rng = np.random.default_rng(7)
    next_state = model.simulate(QueueState(queue_size=5, own_position=2), dt=0.1)
    assert isinstance(next_state, QueueState)
    assert next_state.queue_size >= 0


def test_backward_compat() -> None:
    model = ModelConfig(sigma=0.0, gamma=0.1, kappa=1.5, A=10.0, T=0.05, dt=0.01, initial_mid=100.0)
    inventory_cfg = InventoryConfig(Q_max=2)
    adverse_cfg = AdverseSelectionConfig(enabled=False, epsilon=0.0)

    baseline = simulate_strategy(
        BidOnlyStrategy(),
        model=model,
        inventory_cfg=inventory_cfg,
        adverse_selection_cfg=adverse_cfg,
        n_paths=16,
        seed=21,
    )
    explicit_disabled = simulate_strategy(
        BidOnlyStrategy(),
        model=model,
        inventory_cfg=inventory_cfg,
        adverse_selection_cfg=adverse_cfg,
        n_paths=16,
        seed=21,
        use_queue_position=False,
    )

    assert np.array_equal(explicit_disabled.pnl_paths, baseline.pnl_paths)
    assert np.array_equal(explicit_disabled.inventory_paths, baseline.inventory_paths)
    assert np.array_equal(explicit_disabled.bid_fill_paths, baseline.bid_fill_paths)
    assert np.array_equal(explicit_disabled.ask_fill_paths, baseline.ask_fill_paths)
