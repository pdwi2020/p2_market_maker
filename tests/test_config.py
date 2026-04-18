from pathlib import Path

from p2.config import default_config_path, ensure_run_directories, load_config, resolve_path


def test_load_config_has_expected_sections() -> None:
    config = load_config(default_config_path())
    assert config.run_name == "default_synthetic"
    assert config.simulation.n_paths == 1000
    assert config.paths.results_dir.is_absolute()
    assert config.paths.lobster_dir.is_absolute()


def test_resolve_path_makes_repo_relative_paths_absolute() -> None:
    resolved = resolve_path("results")
    assert resolved.is_absolute()
    assert resolved.name == "results"


def test_ensure_run_directories_creates_run_path(tmp_path: Path) -> None:
    config = load_config()
    config.paths.results_dir = tmp_path
    run_dir = ensure_run_directories(config)
    assert run_dir == tmp_path / config.run_name
    assert run_dir.exists()
