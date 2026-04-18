from dataclasses import dataclass

import numpy as np

from p2.config import AdverseSelectionConfig, InventoryConfig, ModelConfig
from p2.execution import simulate_strategy


@dataclass(slots=True)
class BidOnlyStrategy:
    name: str = "bid_only"

    def quotes(self, mid_prices: np.ndarray, inventory: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        bid = np.asarray(mid_prices, dtype=float)
        ask = np.asarray(mid_prices, dtype=float) + 1e6
        return bid, ask


def test_bid_side_blocks_at_inventory_limit() -> None:
    result = simulate_strategy(
        BidOnlyStrategy(),
        model=ModelConfig(sigma=0.0, gamma=0.1, kappa=1.5, A=1e6, T=0.05, dt=0.01, initial_mid=100.0),
        inventory_cfg=InventoryConfig(Q_max=1),
        adverse_selection_cfg=AdverseSelectionConfig(enabled=False, epsilon=0.0),
        n_paths=1,
        seed=7,
    )
    assert np.max(result.inventory_paths) == 1
    first_hit = int(np.argmax(result.inventory_paths[0] == 1))
    assert not result.bid_fill_paths[0, first_hit:].any()


def test_adverse_selection_moves_mid_after_bid_fill() -> None:
    result = simulate_strategy(
        BidOnlyStrategy(),
        model=ModelConfig(sigma=0.0, gamma=0.1, kappa=1.5, A=1e6, T=0.03, dt=0.01, initial_mid=100.0),
        inventory_cfg=InventoryConfig(Q_max=3),
        adverse_selection_cfg=AdverseSelectionConfig(enabled=True, epsilon=0.1),
        n_paths=1,
        seed=11,
    )
    assert np.allclose(result.mid_price_paths[0], np.array([100.0, 99.9, 99.8, 99.7]))
