"""Torch-backed parameter sweep for AVS quoting sensitivity."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from p2.config import P2Config, ensure_run_directories, load_config


def _detect_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _optimal_half_spread(gamma: float, sigma: float, kappa: float, tau: float) -> float:
    if abs(gamma) < 1e-12:
        return 1.0 / kappa
    return 0.5 * gamma * sigma * sigma * tau + np.log1p(gamma / kappa) / gamma


def sweep_parameters(config: P2Config) -> pd.DataFrame:
    resolved_device = _detect_device(config.sweep.device)
    torch_device = torch.device(resolved_device)

    rows: list[dict[str, float | str]] = []
    grid = list(product(config.sweep.gamma_grid, config.sweep.sigma_grid, config.sweep.T_grid))
    for gamma, sigma, horizon in tqdm(grid, desc="sweep", leave=False):
        n_steps = max(int(round(horizon / config.model.dt)), 1)
        dt = horizon / n_steps
        sqrt_dt = float(np.sqrt(dt))
        n_paths = config.sweep.n_paths

        mid = torch.full((n_paths,), config.model.initial_mid, dtype=torch.float32, device=torch_device)
        inventory = torch.zeros((n_paths,), dtype=torch.int32, device=torch_device)
        cash = torch.zeros((n_paths,), dtype=torch.float32, device=torch_device)
        fill_bid = torch.zeros((n_paths,), dtype=torch.int32, device=torch_device)
        fill_ask = torch.zeros((n_paths,), dtype=torch.int32, device=torch_device)
        abs_inventory = torch.zeros((n_paths,), dtype=torch.float32, device=torch_device)
        spread_realized = torch.zeros((n_paths,), dtype=torch.float32, device=torch_device)

        generator = torch.Generator(device=resolved_device)
        generator.manual_seed(config.sweep.seed + int(round(gamma * 1_000)) + int(round(sigma * 100)) + int(round(horizon * 10)))

        for step in range(n_steps):
            t = step * dt
            tau = max(horizon - t, 0.0)
            half_spread = _optimal_half_spread(gamma=gamma, sigma=sigma, kappa=config.model.kappa, tau=tau)
            reservation = mid - inventory.to(torch.float32) * gamma * sigma * sigma * tau
            bid = reservation - half_spread
            ask = reservation + half_spread

            bid_blocked = inventory >= config.inventory.Q_max
            ask_blocked = inventory <= -config.inventory.Q_max
            delta_bid = torch.where(bid_blocked, torch.inf, torch.clamp(mid - bid, min=0.0))
            delta_ask = torch.where(ask_blocked, torch.inf, torch.clamp(ask - mid, min=0.0))

            lambda_bid = config.model.A * torch.exp(-config.model.kappa * delta_bid)
            lambda_ask = config.model.A * torch.exp(-config.model.kappa * delta_ask)
            bid_prob = torch.where(bid_blocked, 0.0, 1.0 - torch.exp(-lambda_bid * dt))
            ask_prob = torch.where(ask_blocked, 0.0, 1.0 - torch.exp(-lambda_ask * dt))

            bid_fill = torch.rand((n_paths,), generator=generator, device=torch_device) < bid_prob
            ask_fill = torch.rand((n_paths,), generator=generator, device=torch_device) < ask_prob

            cash = cash - bid_fill.to(torch.float32) * bid + ask_fill.to(torch.float32) * ask
            inventory = inventory + bid_fill.to(torch.int32) - ask_fill.to(torch.int32)
            fill_bid = fill_bid + bid_fill.to(torch.int32)
            fill_ask = fill_ask + ask_fill.to(torch.int32)
            abs_inventory = abs_inventory + inventory.abs().to(torch.float32)
            spread_realized = spread_realized + 0.5 * (ask - bid) * (bid_fill.to(torch.float32) + ask_fill.to(torch.float32))

            adverse_shift = 0.0
            if config.adverse_selection.enabled and config.adverse_selection.epsilon > 0.0:
                adverse_shift = (
                    ask_fill.to(torch.float32) * config.adverse_selection.epsilon
                    - bid_fill.to(torch.float32) * config.adverse_selection.epsilon
                )
            diffusion = sigma * sqrt_dt * torch.randn((n_paths,), generator=generator, device=torch_device)
            mid = mid + diffusion + adverse_shift

        pnl = cash + inventory.to(torch.float32) * mid
        pnl_cpu = pnl.detach().cpu().numpy()
        inventory_cpu = inventory.detach().cpu().numpy().astype(float)
        fills_bid_cpu = fill_bid.detach().cpu().numpy().astype(float)
        fills_ask_cpu = fill_ask.detach().cpu().numpy().astype(float)
        rows.append(
            {
                "device": resolved_device,
                "gamma": gamma,
                "sigma": sigma,
                "T": horizon,
                "mean_pnl": float(np.mean(pnl_cpu)),
                "std_pnl": float(np.std(pnl_cpu, ddof=1)) if pnl_cpu.size > 1 else 0.0,
                "sharpe": float(np.mean(pnl_cpu) / np.std(pnl_cpu, ddof=1)) if pnl_cpu.size > 1 and float(np.std(pnl_cpu, ddof=1)) > 0 else 0.0,
                "avg_abs_inventory": float(abs_inventory.mean().item() / n_steps),
                "inventory_variance": float(np.var(inventory_cpu)),
                "spread_capture": float(spread_realized.mean().item()),
                "avg_bid_fill_rate": float(np.sum(fills_bid_cpu) / (n_steps * dt * n_paths)),
                "avg_ask_fill_rate": float(np.sum(fills_ask_cpu) / (n_steps * dt * n_paths)),
                "half_spread_t0": _optimal_half_spread(gamma=gamma, sigma=sigma, kappa=config.model.kappa, tau=horizon),
            }
        )

    return pd.DataFrame(rows)


def plot_heatmap(results: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    unique_T = sorted(results["T"].unique())
    fig, axes = plt.subplots(1, len(unique_T), figsize=(5 * len(unique_T), 4), squeeze=False)

    for axis, horizon in zip(axes[0], unique_T):
        subset = results[results["T"] == horizon]
        pivot = subset.pivot(index="gamma", columns="sigma", values="sharpe")
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="crest", ax=axis)
        axis.set_title(f"Sharpe, T={horizon}")
        axis.set_xlabel("sigma")
        axis.set_ylabel("gamma")

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_sweep_outputs(config: P2Config, results: pd.DataFrame) -> Path:
    run_dir = ensure_run_directories(config)
    csv_path = run_dir / "cuda_sweep.csv"
    image_path = run_dir / "avs_sweep_heatmap.png"
    results.to_csv(csv_path, index=False)
    plot_heatmap(results, image_path)
    return run_dir


def main(config_path: str | Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    results = sweep_parameters(config)
    output_dir = write_sweep_outputs(config, results)
    print(f"saved_results={output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the P2 parameter sweep.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
