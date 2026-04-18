"""Calibration helpers for the simplified LOBSTER replay path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p2.config import P2Config, ensure_run_directories, load_config, resolve_path
from p2.lobster_replay import LOBSTERReplayer
from p2.queue import QueueReactiveModel, calibrate_queue_reactive


DEFAULT_QUEUE_DEPTH_GRID = [1, 5, 10, 20, 50, 100, 200, 500]


def _resolve_replay_files(config: P2Config) -> tuple[Path, Path]:
    if config.replay.orderbook_file and config.replay.message_file:
        return resolve_path(config.replay.orderbook_file), resolve_path(config.replay.message_file)

    lobster_dir = config.paths.lobster_dir
    orderbooks = sorted(lobster_dir.glob("*orderbook*.csv"))
    messages = sorted(lobster_dir.glob("*message*.csv"))
    if not orderbooks or not messages:
        raise FileNotFoundError(
            f"No LOBSTER replay files found in {lobster_dir}. "
            "Populate data/lobster or set replay.orderbook_file and replay.message_file."
        )
    return orderbooks[0], messages[0]


def _event_horizon(times: np.ndarray) -> float:
    if times.size < 2:
        return float(max(times.size, 1))
    horizon = float(times[-1] - times[0])
    return horizon if horizon > 0 else float(times.size)


def calibrate_sigma(orderbook: pd.DataFrame, messages: pd.DataFrame) -> float:
    mid = 0.5 * (orderbook["ask_price_1"].astype(float) + orderbook["bid_price_1"].astype(float))
    times = messages["time"].to_numpy(dtype=float)[: len(mid)]
    returns = np.diff(mid.to_numpy(dtype=float))
    if returns.size == 0:
        return 0.0
    dt = _event_horizon(times) / max(len(returns), 1)
    return float(np.std(returns, ddof=1) / np.sqrt(max(dt, 1e-12)))


def calibrate_arrival_params(orderbook: pd.DataFrame, messages: pd.DataFrame) -> tuple[float, float]:
    n_rows = min(len(orderbook), len(messages))
    if n_rows == 0:
        return 0.0, 1.0
    mid = 0.5 * (
        orderbook["ask_price_1"].to_numpy(dtype=float)[:n_rows]
        + orderbook["bid_price_1"].to_numpy(dtype=float)[:n_rows]
    )
    prices = messages["price"].to_numpy(dtype=float)[:n_rows]
    event_type = messages["event_type"].to_numpy(dtype=int)[:n_rows]
    times = messages["time"].to_numpy(dtype=float)[:n_rows]
    trade_mask = np.isin(event_type, [4, 5])
    deltas = np.abs(prices[trade_mask] - mid[trade_mask])
    if deltas.size == 0:
        return 0.0, 1.0

    horizon = _event_horizon(times)
    bins = np.round(deltas, 2)
    unique_bins, counts = np.unique(bins, return_counts=True)
    intensity = counts / max(horizon, 1e-12)
    positive_mask = intensity > 0
    unique_bins = unique_bins[positive_mask]
    intensity = intensity[positive_mask]
    if unique_bins.size < 2:
        return float(intensity[0]) if intensity.size else 0.0, 1.0

    slope, intercept = np.polyfit(unique_bins, np.log(intensity), deg=1)
    A = float(np.exp(intercept))
    kappa = float(max(-slope, 1e-6))
    return A, kappa


def calibrate_epsilon(orderbook: pd.DataFrame, messages: pd.DataFrame) -> float:
    n_rows = min(len(orderbook), len(messages))
    if n_rows < 2:
        return 0.0
    mid = 0.5 * (
        orderbook["ask_price_1"].to_numpy(dtype=float)[:n_rows]
        + orderbook["bid_price_1"].to_numpy(dtype=float)[:n_rows]
    )
    direction = messages["direction"].to_numpy(dtype=int)[: n_rows - 1]
    next_move = mid[1:] - mid[:-1]
    signed_adverse_move = np.where(direction > 0, next_move, -next_move)
    return float(max(np.mean(np.clip(signed_adverse_move, 0.0, None)), 0.0))


def calibrate_queue_model(
    orderbook: pd.DataFrame,
    messages: pd.DataFrame,
    depth_grid: list[int] | None = None,
) -> QueueReactiveModel:
    events = _queue_calibration_events(orderbook, messages)
    return calibrate_queue_reactive(events, depth_grid or DEFAULT_QUEUE_DEPTH_GRID)


def queue_depth_summary(orderbook: pd.DataFrame) -> dict[str, float]:
    return {
        "queue_depth_mean_bid": float(orderbook["bid_size_1"].astype(float).mean()),
        "queue_depth_mean_ask": float(orderbook["ask_size_1"].astype(float).mean()),
    }


def _queue_calibration_events(orderbook: pd.DataFrame, messages: pd.DataFrame) -> list[dict[str, float]]:
    n_rows = min(len(orderbook), len(messages))
    if n_rows < 2:
        return []

    bid_price = orderbook["bid_price_1"].to_numpy(dtype=float)[:n_rows]
    ask_price = orderbook["ask_price_1"].to_numpy(dtype=float)[:n_rows]
    bid_size = orderbook["bid_size_1"].to_numpy(dtype=int)[:n_rows]
    ask_size = orderbook["ask_size_1"].to_numpy(dtype=int)[:n_rows]
    event_type = messages["event_type"].to_numpy(dtype=int)[:n_rows]
    direction = messages["direction"].to_numpy(dtype=int)[:n_rows]
    price = messages["price"].to_numpy(dtype=float)[:n_rows]
    size = messages["size"].to_numpy(dtype=float)[:n_rows]
    times = messages["time"].to_numpy(dtype=float)[:n_rows]

    records: list[dict[str, float]] = []
    for idx in range(n_rows - 1):
        dt = max(float(times[idx + 1] - times[idx]), 0.0)
        next_event = int(event_type[idx + 1])
        next_direction = int(direction[idx + 1])
        next_price = float(price[idx + 1])
        next_size = max(float(size[idx + 1]), 0.0)

        bid_record = {
            "queue_depth": float(max(int(bid_size[idx]), 0)),
            "dt": dt,
            "arrival_count": 0.0,
            "cancel_count": 0.0,
            "add_count": 0.0,
        }
        ask_record = {
            "queue_depth": float(max(int(ask_size[idx]), 0)),
            "dt": dt,
            "arrival_count": 0.0,
            "cancel_count": 0.0,
            "add_count": 0.0,
        }

        if next_event == 1:
            if next_direction > 0 and np.isclose(next_price, bid_price[idx]):
                bid_record["add_count"] = next_size
            elif next_direction < 0 and np.isclose(next_price, ask_price[idx]):
                ask_record["add_count"] = next_size
        elif next_event in {2, 3}:
            if next_direction > 0 and np.isclose(next_price, bid_price[idx]):
                bid_record["cancel_count"] = next_size
            elif next_direction < 0 and np.isclose(next_price, ask_price[idx]):
                ask_record["cancel_count"] = next_size
        elif next_event in {4, 5}:
            if next_direction > 0 and np.isclose(next_price, ask_price[idx]):
                ask_record["arrival_count"] = next_size
            elif next_direction < 0 and np.isclose(next_price, bid_price[idx]):
                bid_record["arrival_count"] = next_size

        records.append(bid_record)
        records.append(ask_record)

    return records


def calibrate_from_replay_files(orderbook_file: str | Path, message_file: str | Path) -> dict[str, float | int | str]:
    replayer = LOBSTERReplayer().load(orderbook_file=orderbook_file, message_file=message_file)
    assert replayer.orderbook is not None
    assert replayer.messages is not None

    sigma = calibrate_sigma(replayer.orderbook, replayer.messages)
    A, kappa = calibrate_arrival_params(replayer.orderbook, replayer.messages)
    epsilon = calibrate_epsilon(replayer.orderbook, replayer.messages)
    queue_events = _queue_calibration_events(replayer.orderbook, replayer.messages)
    queue_model = calibrate_queue_reactive(queue_events, DEFAULT_QUEUE_DEPTH_GRID)
    queue_summary = queue_depth_summary(replayer.orderbook)
    total_exposure = float(sum(event["dt"] for event in queue_events))
    total_cancel = float(sum(event["cancel_count"] for event in queue_events))
    return {
        "sigma": sigma,
        "A": A,
        "kappa": kappa,
        "epsilon": epsilon,
        **queue_summary,
        "queue_depth_grid": queue_model.depth_grid.tolist() if queue_model.depth_grid is not None else [],
        "queue_arrival_rates": queue_model.arrival_rates.tolist() if queue_model.arrival_rates is not None else [],
        "queue_cancel_rates": queue_model.cancel_rates.tolist() if queue_model.cancel_rates is not None else [],
        "queue_fill_intensities": queue_model.fill_intensities.tolist() if queue_model.fill_intensities is not None else [],
        "queue_cancellation_rate": total_cancel / max(total_exposure, 1e-12),
        "orderbook_file": str(Path(orderbook_file)),
        "message_file": str(Path(message_file)),
        "n_rows": int(min(len(replayer.orderbook), len(replayer.messages))),
    }


def write_calibration(config: P2Config, calibration: dict[str, float | int | str]) -> Path:
    run_dir = ensure_run_directories(config)
    output_path = run_dir / "calibration.json"
    output_path.write_text(json.dumps(calibration, indent=2))
    return output_path


def main(config_path: str | Path | None = None) -> dict[str, float | int | str]:
    try:
        config = load_config(config_path)
        orderbook_file, message_file = _resolve_replay_files(config)
        calibration = calibrate_from_replay_files(orderbook_file, message_file)
        write_calibration(config, calibration)
        return calibration
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate simplified LOBSTER replay parameters.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
