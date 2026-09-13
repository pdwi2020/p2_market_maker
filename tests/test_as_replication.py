from pathlib import Path

import numpy as np

from p2.config import load_config
from p2.simulator import run_as_2008_replication


def test_as_2008_replication_parameters_and_comparison() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "as_2008_replication.yaml")

    assert config.model.initial_mid == 100.0
    assert config.model.T == 1.0
    assert config.model.sigma == 2.0
    assert config.model.dt == 0.005
    assert config.model.A == 140.0
    assert config.model.kappa == 1.5
    assert config.sweep.gamma_grid == [0.01, 0.1, 0.5]
    assert config.simulation.n_paths == 1000
    assert config.inventory.enabled is False
    assert config.adverse_selection.enabled is False

    comparison = run_as_2008_replication(config)

    assert len(comparison) == 6
    assert set(comparison["strategy"]) == {"inventory", "symmetric"}
    assert np.isfinite(comparison.select_dtypes(include="number").to_numpy()).all()
    for gamma in config.sweep.gamma_grid:
        rows = comparison[comparison["gamma"] == gamma].set_index("strategy")
        assert rows.loc["inventory", "mean_profit"] != rows.loc["symmetric", "mean_profit"]
        assert rows.loc["inventory", "std_final_inventory"] != rows.loc["symmetric", "std_final_inventory"]
