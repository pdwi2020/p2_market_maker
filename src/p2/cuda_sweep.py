"""CPU-first parameter sweep with optional accelerator support."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from p2.config import P2Config, ensure_run_directories, load_config

try:
    import torch
except ModuleNotFoundError:  # Optional accelerator dependency.
    torch = None


def _detect_device(device: str) -> str:
    if device == "cpu":
        return device
    if torch is None:
        if device == "auto":
            return "cpu"
        raise RuntimeError("Install the 'gpu' extra to request an accelerator device.")
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


def _grid(config: P2Config) -> list[tuple[float, float, float]]:
    return list(product(config.sweep.gamma_grid, config.sweep.sigma_grid, config.sweep.T_grid))


def _row(
    *,
    device: str,
    gamma: float,
    sigma: float,
    horizon: float,
    pnl: np.ndarray,
    inventory: np.ndarray,
    fill_bid: np.ndarray,
    fill_ask: np.ndarray,
    abs_inventory: float,
    spread_realized: float,
    n_steps: int,
    dt: float,
    n_paths: int,
    kappa: float,
) -> dict[str, float | str]:
    pnl_std = float(np.std(pnl, ddof=1)) if pnl.size > 1 else 0.0
    return {
        "device": device,
        "gamma": gamma,
        "sigma": sigma,
        "T": horizon,
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": pnl_std,
        "sharpe": float(np.mean(pnl) / pnl_std) if pnl_std > 0 else 0.0,
        "avg_abs_inventory": abs_inventory / n_steps,
        "inventory_variance": float(np.var(inventory)),
        "spread_capture": spread_realized,
        "avg_bid_fill_rate": float(np.sum(fill_bid) / (n_steps * dt * n_paths)),
        "avg_ask_fill_rate": float(np.sum(fill_ask) / (n_steps * dt * n_paths)),
        "half_spread_t0": _optimal_half_spread(
            gamma=gamma,
            sigma=sigma,
            kappa=kappa,
            tau=horizon,
        ),
    }


def _sweep_parameters_numpy(config: P2Config) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for gamma, sigma, horizon in tqdm(_grid(config), desc="sweep", leave=False):
        n_steps = max(int(round(horizon / config.model.dt)), 1)
        dt = horizon / n_steps
        sqrt_dt = float(np.sqrt(dt))
        n_paths = config.sweep.n_paths
        seed = config.sweep.seed + int(round(gamma * 1_000)) + int(round(sigma * 100)) + int(round(horizon * 10))
        rng = np.random.default_rng(seed)

        mid = np.full(n_paths, config.model.initial_mid, dtype=float)
        inventory = np.zeros(n_paths, dtype=np.int32)
        cash = np.zeros(n_paths, dtype=float)
        fill_bid = np.zeros(n_paths, dtype=np.int32)
        fill_ask = np.zeros(n_paths, dtype=np.int32)
        abs_inventory = np.zeros(n_paths, dtype=float)
        spread_realized = np.zeros(n_paths, dtype=float)

        for step in range(n_steps):
            t = step * dt
            tau = max(horizon - t, 0.0)
            half_spread = _optimal_half_spread(gamma=gamma, sigma=sigma, kappa=config.model.kappa, tau=tau)
            reservation = mid - inventory * gamma * sigma * sigma * tau
            bid = reservation - half_spread
            ask = reservation + half_spread

            bid_blocked = inventory >= config.inventory.Q_max
            ask_blocked = inventory <= -config.inventory.Q_max
            delta_bid = np.where(bid_blocked, np.inf, np.maximum(mid - bid, 0.0))
            delta_ask = np.where(ask_blocked, np.inf, np.maximum(ask - mid, 0.0))

            lambda_bid = config.model.A * np.exp(-config.model.kappa * delta_bid)
            lambda_ask = config.model.A * np.exp(-config.model.kappa * delta_ask)
            bid_prob = np.where(bid_blocked, 0.0, 1.0 - np.exp(-lambda_bid * dt))
            ask_prob = np.where(ask_blocked, 0.0, 1.0 - np.exp(-lambda_ask * dt))

            bid_fill_step = rng.random(n_paths) < bid_prob
            ask_fill_step = rng.random(n_paths) < ask_prob

            cash = cash - bid_fill_step * bid + ask_fill_step * ask
            inventory = inventory + bid_fill_step.astype(np.int32) - ask_fill_step.astype(np.int32)
            fill_bid = fill_bid + bid_fill_step.astype(np.int32)
            fill_ask = fill_ask + ask_fill_step.astype(np.int32)
            abs_inventory = abs_inventory + np.abs(inventory)
            spread_realized = spread_realized + 0.5 * (ask - bid) * (bid_fill_step + ask_fill_step)

            adverse_shift = 0.0
            if config.adverse_selection.enabled and config.adverse_selection.epsilon > 0.0:
                adverse_shift = (
                    ask_fill_step * config.adverse_selection.epsilon
                    - bid_fill_step * config.adverse_selection.epsilon
                )
            diffusion = sigma * sqrt_dt * rng.standard_normal(n_paths)
            mid = mid + diffusion + adverse_shift

        rows.append(_row(
            device="cpu",
            gamma=gamma,
            sigma=sigma,
            horizon=horizon,
            pnl=cash + inventory * mid,
            inventory=inventory,
            fill_bid=fill_bid,
            fill_ask=fill_ask,
            abs_inventory=float(np.mean(abs_inventory)),
            spread_realized=float(np.mean(spread_realized)),
            n_steps=n_steps,
            dt=dt,
            n_paths=n_paths,
            kappa=config.model.kappa,
        ))

    return pd.DataFrame(rows)


def _sweep_parameters_torch(config: P2Config, device: str) -> pd.DataFrame:
    if torch is None:  # Defensive guard for direct calls.
        raise RuntimeError("Install the 'gpu' extra to use an accelerator device.")

    torch_device = torch.device(device)
    rows: list[dict[str, float | str]] = []
    for gamma, sigma, horizon in tqdm(_grid(config), desc="sweep", leave=False):
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

        generator = torch.Generator(device=device)
        generator.manual_seed(
            config.sweep.seed
            + int(round(gamma * 1_000))
            + int(round(sigma * 100))
            + int(round(horizon * 10))
        )

        for step in range(n_steps):
            tau = max(horizon - step * dt, 0.0)
            half_spread = _optimal_half_spread(gamma, sigma, config.model.kappa, tau)
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

            bid_fill_step = torch.rand((n_paths,), generator=generator, device=torch_device) < bid_prob
            ask_fill_step = torch.rand((n_paths,), generator=generator, device=torch_device) < ask_prob
            cash = cash - bid_fill_step.to(torch.float32) * bid + ask_fill_step.to(torch.float32) * ask
            inventory = inventory + bid_fill_step.to(torch.int32) - ask_fill_step.to(torch.int32)
            fill_bid = fill_bid + bid_fill_step.to(torch.int32)
            fill_ask = fill_ask + ask_fill_step.to(torch.int32)
            abs_inventory = abs_inventory + inventory.abs().to(torch.float32)
            spread_realized = spread_realized + 0.5 * (ask - bid) * (
                bid_fill_step.to(torch.float32) + ask_fill_step.to(torch.float32)
            )

            adverse_shift = 0.0
            if config.adverse_selection.enabled and config.adverse_selection.epsilon > 0.0:
                adverse_shift = (
                    ask_fill_step.to(torch.float32) * config.adverse_selection.epsilon
                    - bid_fill_step.to(torch.float32) * config.adverse_selection.epsilon
                )
            diffusion = sigma * sqrt_dt * torch.randn((n_paths,), generator=generator, device=torch_device)
            mid = mid + diffusion + adverse_shift

        rows.append(_row(
            device=device,
            gamma=gamma,
            sigma=sigma,
            horizon=horizon,
            pnl=(cash + inventory.to(torch.float32) * mid).detach().cpu().numpy(),
            inventory=inventory.detach().cpu().numpy(),
            fill_bid=fill_bid.detach().cpu().numpy(),
            fill_ask=fill_ask.detach().cpu().numpy(),
            abs_inventory=float(abs_inventory.mean().item()),
            spread_realized=float(spread_realized.mean().item()),
            n_steps=n_steps,
            dt=dt,
            n_paths=n_paths,
            kappa=config.model.kappa,
        ))

    return pd.DataFrame(rows)


def sweep_parameters(config: P2Config) -> pd.DataFrame:
    resolved_device = _detect_device(config.sweep.device)
    if resolved_device == "cpu":
        return _sweep_parameters_numpy(config)
    return _sweep_parameters_torch(config, resolved_device)


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
