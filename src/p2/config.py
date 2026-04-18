"""Typed configuration helpers for P2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parents[2]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_dir: Path = Path("results")
    data_dir: Path = Path("data")
    lobster_dir: Path = Path("data/lobster")


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sigma: float = 1.0
    gamma: float = 0.1
    kappa: float = 1.5
    A: float = 140.0
    T: float = 1.0
    dt: float = 0.001
    initial_mid: float = 100.0


class InventoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Q_max: int = Field(default=10, ge=1)


class AdverseSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    epsilon: float = Field(default=0.02, ge=0.0)


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_paths: int = Field(default=1000, ge=1)
    seed: int = 42
    backend: str = "numpy"


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "auto"
    gamma_grid: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.1, 0.5, 1.0])
    sigma_grid: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0])
    T_grid: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0])
    n_paths: int = Field(default=2000, ge=1)
    seed: int = 123


class ReplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orderbook_file: str | None = None
    message_file: str | None = None
    use_calibrated_params: bool = False


class QueueModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    arrival_rate_depth_coeff: float = Field(default=0.05, ge=0.0)
    cancel_rate_depth_coeff: float = Field(default=0.02, ge=0.0)
    fill_intensity_depth_coeff: float = Field(default=0.01, ge=0.0)


class P2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str = "default_synthetic"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    inventory: InventoryConfig = Field(default_factory=InventoryConfig)
    adverse_selection: AdverseSelectionConfig = Field(default_factory=AdverseSelectionConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    queue_model: QueueModelConfig = Field(default_factory=QueueModelConfig)

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "P2Config":
        self.paths.results_dir = resolve_path(self.paths.results_dir)
        self.paths.data_dir = resolve_path(self.paths.data_dir)
        self.paths.lobster_dir = resolve_path(self.paths.lobster_dir)
        return self

    def results_path(self, *parts: str) -> Path:
        base = self.paths.results_dir / self.run_name
        return base.joinpath(*parts)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "p2_base.yaml"


def load_config(path: str | Path | None = None) -> P2Config:
    config_path = resolve_path(path) if path is not None else default_config_path()
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}
    return P2Config.model_validate(raw)


def ensure_run_directories(config: P2Config) -> Path:
    run_dir = config.results_path()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
