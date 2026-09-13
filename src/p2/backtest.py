"""Backtesting helpers for replaying simplified LOBSTER data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p2.calibration import calibrate_from_replay_files
from p2.config import P2Config, ensure_run_directories, load_config, resolve_path
from p2.hjb_solver import optimal_ask, optimal_bid
from p2.lobster_replay import BacktestResult, LOBSTERReplayer


def _find_data_file(data_dir: Path, needle: str) -> Path:
    matches = sorted(data_dir.glob(f"*{needle}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No LOBSTER file matching '*{needle}*.csv' found in {data_dir}")
    return matches[0]


def _resolve_files(config: P2Config) -> tuple[Path, Path]:
    if config.replay.orderbook_file and config.replay.message_file:
        return resolve_path(config.replay.orderbook_file), resolve_path(config.replay.message_file)
    try:
        return _find_data_file(config.paths.lobster_dir, "orderbook"), _find_data_file(config.paths.lobster_dir, "message")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"No LOBSTER replay files found in {config.paths.lobster_dir}. "
            "Populate data/lobster or set replay.orderbook_file and replay.message_file."
        ) from error


def run_lobster_backtest(config: P2Config) -> BacktestResult:
    orderbook_file, message_file = _resolve_files(config)
    gamma = config.model.gamma
    sigma = config.model.sigma
    kappa = config.model.kappa
    T = config.replay.session_duration_seconds

    if config.replay.use_calibrated_params:
        calibrated = calibrate_from_replay_files(orderbook_file, message_file)
        sigma = float(calibrated["sigma"])
        kappa = float(calibrated["kappa"])

    replayer = LOBSTERReplayer().load(orderbook_file=orderbook_file, message_file=message_file)

    def strategy(mid_price: float, inventory: int, t: float) -> tuple[float, float]:
        return (
            optimal_bid(mid_price, inventory, t, T, gamma, sigma, kappa),
            optimal_ask(mid_price, inventory, t, T, gamma, sigma, kappa),
        )

    return replayer.run_strategy(
        strategy,
        use_queue_position=config.queue_model.enabled,
        session_duration_seconds=T,
        post_only_mode=config.replay.post_only_mode,
        order_size=config.queue_model.order_size,
        cancellation_rule=config.queue_model.cancellation_rule,
        latency_ms=config.replay.latency_ms,
        inventory_limit=config.inventory.Q_max,
        maker_rebate_per_share=config.replay.maker_rebate_per_share,
        taker_fee_per_share=config.replay.taker_fee_per_share,
    )


def pnl_attribution(result: BacktestResult) -> dict[str, float | None]:
    inventory_path = np.asarray(result.inventory_path, dtype=float)
    return {
        "terminal_pnl": float(result.pnl),
        "inventory_variance": float(np.var(inventory_path)),
        "avg_abs_inventory": float(np.mean(np.abs(inventory_path))),
        "quoted_width": float(result.quoted_width),
        "realized_spread": float(result.realized_spread),
        "spread_capture_pct": result.spread_capture_pct,
        "realized_spread_pnl": float(result.realized_spread_pnl),
        "inventory_mtm_pnl": float(result.inventory_mtm_pnl),
        "maker_rebates_pnl": float(result.maker_rebates_pnl),
        "taker_fees_pnl": float(result.taker_fees_pnl),
        "fees_and_rebates_pnl": float(result.fees_and_rebates_pnl),
        "markout_1s": result.markout_1s,
        "markout_5s": result.markout_5s,
        "markout_30s": result.markout_30s,
        "markout_60s": result.markout_60s,
        "n_fills": float(result.fill_quantity),
        "n_fill_events": float(len(result.fills)),
        "marketable_fill_count": float(result.marketable_fill_count),
        "avg_position_at_fill": float(result.avg_position_at_fill),
        "cancellation_rate": float(result.cancellation_rate),
        "time_to_fill_mean": float(result.time_to_fill_mean),
        "time_to_fill_p50": float(result.time_to_fill_p50),
        "time_to_fill_p90": float(result.time_to_fill_p90),
    }


def write_backtest_outputs(config: P2Config, result: BacktestResult) -> Path:
    run_dir = ensure_run_directories(config)
    summary_path = run_dir / "replay_summary.json"
    inventory_path = run_dir / "replay_inventory.csv"
    summary_path.write_text(json.dumps(pnl_attribution(result), indent=2))
    inventory_path.write_text("step,inventory\n" + "\n".join(f"{idx},{value}" for idx, value in enumerate(result.inventory_path)))
    return run_dir


def main(config_path: str | Path | None = None) -> BacktestResult:
    try:
        config = load_config(config_path)
        result = run_lobster_backtest(config)
        write_backtest_outputs(config, result)
        return result
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the simplified LOBSTER replay backtest.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
