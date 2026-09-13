from pathlib import Path

from p2.config import P2Config, REPO_ROOT, default_config_path, ensure_run_directories, load_config, resolve_path


def test_load_config_has_expected_sections() -> None:
    config = load_config(default_config_path())
    assert config.run_name == "default_synthetic"
    assert config.simulation.n_paths == 1000
    assert config.paths.results_dir.is_absolute()
    assert config.paths.data_dir == REPO_ROOT / "data"
    assert config.paths.lobster_dir.is_absolute()
    assert config.paths.lobster_dir == REPO_ROOT / "data" / "lobster"
    assert config.paths.cache_dir == REPO_ROOT / ".cache"
    assert config.paths.lake_dir is None


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


def test_environment_paths_override_local_defaults(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "market-data"
    cache_dir = tmp_path / "cache"
    lake_dir = tmp_path / "lake"
    monkeypatch.setenv("P2_DATA_DIR", str(data_dir))
    monkeypatch.setenv("P2_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("P2_LAKE_DIR", str(lake_dir))

    config = P2Config()

    assert config.paths.data_dir == data_dir
    assert config.paths.lobster_dir == data_dir / "lobster"
    assert config.paths.cache_dir == cache_dir
    assert config.paths.lake_dir == lake_dir


def test_explicit_lobster_path_overrides_data_default(tmp_path: Path) -> None:
    lobster_dir = tmp_path / "custom-lobster"

    config = P2Config.model_validate({"paths": {"lobster_dir": lobster_dir}})

    assert config.paths.lobster_dir == lobster_dir
