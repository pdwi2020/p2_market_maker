import numpy as np

from p2.simulator import AVSSimulator


def _make_simulator(seed: int = 42) -> AVSSimulator:
    return AVSSimulator(sigma=1.0, gamma=0.1, kappa=1.5, A=80.0, T=0.2, dt=0.01, Q_max=4, seed=seed)


def test_pnl_mass_balance() -> None:
    result = _make_simulator().run(n_paths=64)
    assert result.cash_paths is not None
    assert result.mid_price_paths is not None
    terminal_mid = result.mid_price_paths[:, -1]
    terminal_inventory = result.inventory_paths[:, -1]
    assert np.allclose(result.pnl_paths, result.cash_paths + terminal_inventory * terminal_mid)


def test_inventory_bounded() -> None:
    result = _make_simulator().run(n_paths=128)
    assert np.max(np.abs(result.inventory_paths)) <= 4


def test_seed_reproducibility() -> None:
    first = _make_simulator(seed=7).run(n_paths=64)
    second = _make_simulator(seed=7).run(n_paths=64)
    assert np.array_equal(first.pnl_paths, second.pnl_paths)
    assert np.array_equal(first.inventory_paths, second.inventory_paths)


def test_n_paths_shapes() -> None:
    simulator = _make_simulator()
    result = simulator.run(n_paths=32)
    n_steps = int(round(simulator.T / simulator.dt)) + 1
    assert result.pnl_paths.shape == (32,)
    assert result.inventory_paths.shape == (32, n_steps)
    assert result.fill_count_bid.shape == (32,)
    assert result.fill_count_ask.shape == (32,)
