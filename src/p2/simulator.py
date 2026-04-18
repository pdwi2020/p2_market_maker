"""Synthetic Monte Carlo simulator for the Avellaneda-Stoikov market maker."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console

from p2.baselines import ConstantSpreadMM, SymmetricMM, compare_strategies
from p2.config import (
    AdverseSelectionConfig,
    InventoryConfig,
    ModelConfig,
    P2Config,
    QueueModelConfig,
    ensure_run_directories,
    load_config,
)
from p2.execution import AvellanedaStoikovStrategy, SimResult, simulate_strategy


@dataclass(slots=True)
class AVSSimulator:
    sigma: float
    gamma: float
    kappa: float
    A: float
    T: float
    dt: float
    Q_max: int
    seed: int = 42
    initial_mid: float = 100.0
    adverse_selection_enabled: bool = True
    epsilon: float = 0.02
    backend: str = "numpy"
    use_queue_position: bool = False
    queue_cfg: QueueModelConfig | None = None

    def run(self, n_paths: int) -> SimResult:
        if self.backend not in {"numpy", "auto"}:
            raise ValueError(f"Unsupported simulation backend '{self.backend}'. Expected 'numpy' or 'auto'.")
        model = ModelConfig(
            sigma=self.sigma,
            gamma=self.gamma,
            kappa=self.kappa,
            A=self.A,
            T=self.T,
            dt=self.dt,
            initial_mid=self.initial_mid,
        )
        inventory_cfg = InventoryConfig(Q_max=self.Q_max)
        adverse_cfg = AdverseSelectionConfig(
            enabled=self.adverse_selection_enabled,
            epsilon=self.epsilon,
        )
        return simulate_strategy(
            AvellanedaStoikovStrategy(model),
            model=model,
            inventory_cfg=inventory_cfg,
            adverse_selection_cfg=adverse_cfg,
            n_paths=n_paths,
            seed=self.seed,
            use_queue_position=self.use_queue_position,
            queue_cfg=self.queue_cfg,
        )


def _make_avs_from_config(config: P2Config) -> AVSSimulator:
    return AVSSimulator(
        sigma=config.model.sigma,
        gamma=config.model.gamma,
        kappa=config.model.kappa,
        A=config.model.A,
        T=config.model.T,
        dt=config.model.dt,
        Q_max=config.inventory.Q_max,
        seed=config.simulation.seed,
        initial_mid=config.model.initial_mid,
        adverse_selection_enabled=config.adverse_selection.enabled,
        epsilon=config.adverse_selection.epsilon,
        backend=config.simulation.backend,
        use_queue_position=config.queue_model.enabled,
        queue_cfg=config.queue_model,
    )


def pathwise_metrics_frame(result: SimResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path_id": range(result.pnl_paths.shape[0]),
            "terminal_pnl": result.pnl_paths,
            "terminal_inventory": result.inventory_paths[:, -1],
            "bid_fills": result.fill_count_bid,
            "ask_fills": result.fill_count_ask,
        }
    )


def simulation_summary(result: SimResult) -> dict[str, float | int | str]:
    return {
        "strategy": result.strategy,
        "mean_terminal_pnl": result.mean_pnl,
        "std_terminal_pnl": result.std_pnl,
        "sharpe": result.sharpe,
        "avg_abs_inventory": result.avg_abs_inventory,
        "inventory_variance": result.inventory_variance,
        "spread_capture": result.spread_capture,
        "avg_bid_fill_rate": result.avg_bid_fill_rate,
        "avg_ask_fill_rate": result.avg_ask_fill_rate,
        "n_paths": int(result.pnl_paths.shape[0]),
    }


def run_default_experiment(config: P2Config) -> tuple[SimResult, pd.DataFrame]:
    avs = _make_avs_from_config(config)
    symmetric = SymmetricMM(half_spread=0.5 * (avs.kappa / max(avs.kappa, 1.0)))
    constant = ConstantSpreadMM(
        sigma=avs.sigma,
        gamma=avs.gamma,
        kappa=avs.kappa,
        T=avs.T,
    )

    avs_result = avs.run(n_paths=config.simulation.n_paths)
    baseline_comparison = compare_strategies(
        {
            "avs": avs,
            "symmetric": (
                symmetric,
                {
                    "sigma": avs.sigma,
                    "gamma": avs.gamma,
                    "kappa": avs.kappa,
                    "A": avs.A,
                    "T": avs.T,
                    "dt": avs.dt,
                    "Q_max": avs.Q_max,
                    "seed": avs.seed,
                    "s0": avs.initial_mid,
                    "adverse_selection": AdverseSelectionConfig(
                        enabled=avs.adverse_selection_enabled,
                        epsilon=avs.epsilon,
                    ),
                },
            ),
            "constant_spread": (
                constant,
                {
                    "sigma": avs.sigma,
                    "gamma": avs.gamma,
                    "kappa": avs.kappa,
                    "A": avs.A,
                    "T": avs.T,
                    "dt": avs.dt,
                    "Q_max": avs.Q_max,
                    "seed": avs.seed,
                    "s0": avs.initial_mid,
                    "adverse_selection": AdverseSelectionConfig(
                        enabled=avs.adverse_selection_enabled,
                        epsilon=avs.epsilon,
                    ),
                },
            ),
        },
        n_paths=config.simulation.n_paths,
    )
    return avs_result, baseline_comparison


def write_simulation_outputs(config: P2Config, result: SimResult, baseline_comparison: pd.DataFrame) -> Path:
    run_dir = ensure_run_directories(config)
    summary_path = run_dir / "summary.json"
    pathwise_path = run_dir / "pathwise_metrics.csv"
    baseline_path = run_dir / "baseline_comparison.csv"

    summary_payload = simulation_summary(result)
    summary_payload["run_name"] = config.run_name
    summary_payload["baseline_best_strategy"] = str(baseline_comparison.iloc[0]["strategy"])
    summary_payload["baseline_best_sharpe"] = float(baseline_comparison.iloc[0]["sharpe"])

    summary_path.write_text(json.dumps(summary_payload, indent=2))
    pathwise_metrics_frame(result).to_csv(pathwise_path, index=False)
    baseline_comparison.to_csv(baseline_path, index=False)
    return run_dir


def _print_summary(result: SimResult) -> None:
    console = Console()
    console.print("[bold]P2 AVS Simulation Summary[/bold]")
    console.print(f"paths={result.pnl_paths.shape[0]}")
    console.print(f"mean_pnl={result.mean_pnl:.6f}")
    console.print(f"std_pnl={result.std_pnl:.6f}")
    console.print(f"sharpe={result.sharpe:.6f}")
    console.print(f"avg_bid_fill_rate={result.avg_bid_fill_rate:.6f}")
    console.print(f"avg_ask_fill_rate={result.avg_ask_fill_rate:.6f}")
    console.print(f"avg_abs_inventory={result.avg_abs_inventory:.6f}")
    console.print(f"spread_capture={result.spread_capture:.6f}")


def main(config_path: str | Path | None = None) -> SimResult:
    config = load_config(config_path)
    result, baseline_comparison = run_default_experiment(config)
    output_dir = write_simulation_outputs(config, result, baseline_comparison)
    _print_summary(result)
    Console().print(f"saved_results={output_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the default P2 synthetic simulation.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
