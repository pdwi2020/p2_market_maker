from pathlib import Path

import pytest

from p2.config import load_config
from p2 import cuda_sweep
from p2.cuda_sweep import plot_heatmap, sweep_parameters, write_sweep_outputs


def test_sweep_smoke(tmp_path: Path) -> None:
    config = load_config()
    config.paths.results_dir = tmp_path
    config.sweep.device = "cpu"
    config.sweep.gamma_grid = [0.1, 0.5]
    config.sweep.sigma_grid = [0.5]
    config.sweep.T_grid = [0.5]
    config.sweep.n_paths = 32
    config.model.dt = 0.01

    results = sweep_parameters(config)
    assert not results.empty
    run_dir = write_sweep_outputs(config, results)
    assert (run_dir / "cuda_sweep.csv").exists()
    assert (run_dir / "avs_sweep_heatmap.png").exists()

    extra_plot = tmp_path / "extra_heatmap.png"
    plot_heatmap(results, extra_plot)
    assert extra_plot.exists()


def test_sweep_uses_cpu_when_torch_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config()
    config.sweep.device = "auto"
    config.sweep.gamma_grid = [0.1]
    config.sweep.sigma_grid = [0.5]
    config.sweep.T_grid = [0.1]
    config.sweep.n_paths = 8
    config.model.dt = 0.1
    monkeypatch.setattr(cuda_sweep, "torch", None)

    results = sweep_parameters(config)

    assert results["device"].tolist() == ["cpu"]


def test_accelerator_requires_gpu_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_sweep, "torch", None)

    with pytest.raises(RuntimeError, match="gpu"):
        cuda_sweep._detect_device("cuda")
