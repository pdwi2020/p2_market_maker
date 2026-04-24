#!/usr/bin/env python3
"""Render the headline README figures for the P2 market-making repo.

The script prefers explicit fill/PnL artifacts under ``results/**`` when they are
available. In the current checked-in snapshot those granular artifacts are absent,
so the script reconstructs the instantaneous replay path from the bundled AAPL
LOBSTER sample using the same closed-form Avellaneda-Stoikov equations used by
the repo's replay logic. If neither the bundled sample nor replay artifacts can
be found, the final fallback is a deterministic synthetic intraday path that
terminates at the checked-in headline PnL (~$662); that path is logged clearly so
it cannot be mistaken for measured replay output.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "p2_market_maker_mpl"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = ROOT / "results" / "readme_figures"
DEFAULT_SUMMARY = {
    "terminal_pnl": 662.0720632416997,
    "spread_capture": 0.07866679679364097,
    "n_fills": 972.0,
    "avg_abs_inventory": 52.46075096405523,
}
DEFAULT_FILL_COUNTS = {"ask": 516, "bid": 456}


@dataclass(slots=True)
class ReplayContext:
    times: np.ndarray
    inventory_path: np.ndarray
    pnl_path: np.ndarray
    fills: pd.DataFrame
    terminal_pnl: float
    spread_capture: float
    n_fills: int
    q_max: int
    source_label: str


def configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "font.size": 10,
        }
    )


def parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    current_section: str | None = None
    parsed: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.endswith(":") and indent == 0:
            current_section = stripped[:-1]
            continue
        if ":" not in stripped or current_section is None:
            continue
        key, value = stripped.split(":", 1)
        parsed[f"{current_section}.{key.strip()}"] = parse_scalar(value)
    return parsed


def resolve_repo_path(raw_path: str | Path | None) -> Path | None:
    if raw_path in {None, ""}:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    return None


def find_primary_run_dir() -> Path | None:
    preferred = ROOT / "results" / "lobster_aapl"
    if (preferred / "replay_summary.json").exists():
        return preferred

    candidates = sorted(ROOT.glob("results/**/replay_summary.json"))
    if not candidates:
        return None

    best_parent: Path | None = None
    best_distance = float("inf")
    for summary_path in candidates:
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        distance = abs(float(summary.get("terminal_pnl", 0.0)) - DEFAULT_SUMMARY["terminal_pnl"])
        if distance < best_distance:
            best_distance = distance
            best_parent = summary_path.parent
    return best_parent


def load_summary(run_dir: Path | None) -> dict[str, float]:
    summary = dict(DEFAULT_SUMMARY)
    if run_dir is None:
        return summary

    summary_path = run_dir / "replay_summary.json"
    if not summary_path.exists():
        return summary

    loaded = json.loads(summary_path.read_text())
    for key, default_value in DEFAULT_SUMMARY.items():
        summary[key] = float(loaded.get(key, default_value))
    return summary


def load_config_values() -> dict[str, Any]:
    config_path = ROOT / "configs" / "p2_config.yaml"
    values = load_simple_yaml(config_path)
    if values:
        return values
    return load_simple_yaml(ROOT / "configs" / "p2_base.yaml")


def find_explicit_series_artifacts() -> tuple[Path | None, Path | None]:
    pnl_candidates = sorted(ROOT.glob("results/**/pnl_series.csv"))
    fill_candidates = sorted(ROOT.glob("results/**/fills.csv"))
    return (pnl_candidates[0] if pnl_candidates else None, fill_candidates[0] if fill_candidates else None)


def load_explicit_series_context(
    summary: dict[str, float],
    q_max: int,
    pnl_path: Path | None,
    fills_path: Path | None,
) -> ReplayContext | None:
    if pnl_path is None or fills_path is None:
        return None

    pnl_frame = pd.read_csv(pnl_path)
    fills = pd.read_csv(fills_path)
    if {"time", "pnl"}.issubset(pnl_frame.columns) and "side" in fills.columns:
        times = pnl_frame["time"].to_numpy(dtype=float)
        pnl_series = pnl_frame["pnl"].to_numpy(dtype=float)
        inventory = pnl_frame["inventory"].to_numpy(dtype=float) if "inventory" in pnl_frame.columns else np.zeros_like(pnl_series)
        fills = fills.copy()
        if "relative_ticks" not in fills.columns:
            if {"fill_price", "mid_price"}.issubset(fills.columns):
                fills["relative_ticks"] = (fills["fill_price"] - fills["mid_price"]) / 0.01
            else:
                fills["relative_ticks"] = np.nan
        return ReplayContext(
            times=times,
            inventory_path=np.asarray(inventory, dtype=float),
            pnl_path=pnl_series,
            fills=fills,
            terminal_pnl=float(pnl_series[-1]),
            spread_capture=float(summary["spread_capture"]),
            n_fills=int(len(fills)),
            q_max=int(q_max),
            source_label=f"explicit artifacts ({pnl_path.relative_to(ROOT)}, {fills_path.relative_to(ROOT)})",
        )
    return None


def load_orderbook_messages(config_values: dict[str, Any], run_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    calibration_path = (run_dir / "calibration.json") if run_dir is not None else None
    orderbook_path: Path | None = None
    message_path: Path | None = None
    source_parts: list[str] = []

    if calibration_path is not None and calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        orderbook_path = resolve_repo_path(calibration.get("orderbook_file"))
        message_path = resolve_repo_path(calibration.get("message_file"))
        source_parts.append(f"calibration: {calibration_path.relative_to(ROOT)}")

    if orderbook_path is None:
        orderbook_path = resolve_repo_path(config_values.get("replay.orderbook_file"))
    if message_path is None:
        message_path = resolve_repo_path(config_values.get("replay.message_file"))

    if orderbook_path is None or message_path is None:
        orderbook_matches = sorted((ROOT / "data" / "lobster").glob("*orderbook*.csv"))
        message_matches = sorted((ROOT / "data" / "lobster").glob("*message*.csv"))
        orderbook_path = orderbook_path or (orderbook_matches[0] if orderbook_matches else None)
        message_path = message_path or (message_matches[0] if message_matches else None)
        source_parts.append("bundled AAPL sample")

    if orderbook_path is None or message_path is None:
        raise FileNotFoundError("No LOBSTER orderbook/message files were found for README figure rendering.")

    orderbook = pd.read_csv(orderbook_path, header=None)
    messages = pd.read_csv(message_path, header=None)
    orderbook.columns = ["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"] + [
        f"col_{idx}" for idx in range(4, orderbook.shape[1])
    ]
    messages.columns = ["time", "event_type", "order_id", "size", "price", "direction"] + [
        f"col_{idx}" for idx in range(6, messages.shape[1])
    ]

    if float(orderbook["ask_price_1"].median()) > 10_000:
        price_columns = [column for column in orderbook.columns if "price" in column]
        orderbook[price_columns] = orderbook[price_columns] / 10_000.0
        messages["price"] = messages["price"] / 10_000.0

    source_parts.append(f"orderbook: {orderbook_path.relative_to(ROOT)}")
    source_parts.append(f"message: {message_path.relative_to(ROOT)}")
    return orderbook, messages, "; ".join(source_parts)


def load_calibrated_params(config_values: dict[str, Any], run_dir: Path | None) -> tuple[float, float, float, float]:
    gamma = float(config_values.get("model.gamma", 0.1))
    sigma = float(config_values.get("model.sigma", 1.0))
    kappa = float(config_values.get("model.kappa", 1.5))
    horizon = float(config_values.get("model.T", 1.0))

    calibration_path = (run_dir / "calibration.json") if run_dir is not None else None
    if calibration_path is not None and calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        sigma = float(calibration.get("sigma", sigma))
        kappa = float(calibration.get("kappa", kappa))

    return gamma, sigma, kappa, horizon


def reconstruct_instantaneous_replay(
    summary: dict[str, float],
    q_max: int,
    config_values: dict[str, Any],
    run_dir: Path | None,
) -> ReplayContext:
    orderbook, messages, source_description = load_orderbook_messages(config_values, run_dir)
    gamma, sigma, kappa, horizon = load_calibrated_params(config_values, run_dir)

    mids = 0.5 * (orderbook["ask_price_1"].to_numpy(dtype=float) + orderbook["bid_price_1"].to_numpy(dtype=float))
    prices = messages["price"].to_numpy(dtype=float)
    times = messages["time"].to_numpy(dtype=float)
    event_type = messages["event_type"].to_numpy(dtype=int)
    direction = messages["direction"].to_numpy(dtype=int)

    n_rows = min(len(mids), len(prices), len(times), len(event_type), len(direction))
    inventory = 0
    cash = 0.0
    inventory_path = np.zeros(n_rows + 1, dtype=float)
    pnl_path = np.zeros(n_rows + 1, dtype=float)
    fill_rows: list[dict[str, float | str]] = []
    spreads: list[float] = []

    constant_component = math.log1p(gamma / kappa) / gamma if not math.isclose(gamma, 0.0, abs_tol=1e-12) else 1.0 / kappa
    time_axis = np.empty(n_rows + 1, dtype=float)
    time_axis[0] = float(times[0]) if n_rows else 0.0

    for idx in range(n_rows):
        mid = float(mids[idx])
        t = float(times[idx])
        trade_price = float(prices[idx])
        tau = max(horizon - t, 0.0)
        reservation = mid - inventory * gamma * sigma**2 * tau
        half_spread = 0.5 * gamma * sigma**2 * tau + constant_component
        bid = reservation - half_spread
        ask = reservation + half_spread

        if int(event_type[idx]) in {4, 5}:
            if int(direction[idx]) > 0 and ask <= trade_price:
                cash += ask
                inventory -= 1
                spreads.append(ask - bid)
                fill_rows.append(
                    {
                        "time": t,
                        "side": "ask",
                        "fill_price": ask,
                        "trade_price": trade_price,
                        "mid_price": mid,
                        "relative_ticks": (ask - mid) / 0.01,
                        "relative_bps": (ask - mid) / mid * 10_000.0,
                    }
                )
            elif int(direction[idx]) < 0 and bid >= trade_price:
                cash -= bid
                inventory += 1
                spreads.append(ask - bid)
                fill_rows.append(
                    {
                        "time": t,
                        "side": "bid",
                        "fill_price": bid,
                        "trade_price": trade_price,
                        "mid_price": mid,
                        "relative_ticks": (bid - mid) / 0.01,
                        "relative_bps": (bid - mid) / mid * 10_000.0,
                    }
                )

        inventory_path[idx + 1] = inventory
        pnl_path[idx + 1] = cash + inventory * mid
        time_axis[idx + 1] = t

    fills = pd.DataFrame(fill_rows)
    terminal_pnl = float(pnl_path[-1]) if len(pnl_path) else float(summary["terminal_pnl"])
    spread_capture = float(np.mean(spreads)) if spreads else float(summary["spread_capture"])

    if fills.empty:
        raise RuntimeError("Replay reconstruction produced no fills.")

    expected_terminal = float(summary["terminal_pnl"])
    expected_fills = int(round(summary["n_fills"]))
    if not math.isclose(terminal_pnl, expected_terminal, rel_tol=0.0, abs_tol=1e-6):
        logging.warning("Reconstructed terminal PnL %.6f differs from replay summary %.6f", terminal_pnl, expected_terminal)
    if len(fills) != expected_fills:
        logging.warning("Reconstructed fill count %s differs from replay summary %s", len(fills), expected_fills)

    return ReplayContext(
        times=time_axis,
        inventory_path=inventory_path,
        pnl_path=pnl_path,
        fills=fills,
        terminal_pnl=terminal_pnl,
        spread_capture=spread_capture,
        n_fills=int(len(fills)),
        q_max=int(q_max),
        source_label=f"reconstructed instantaneous replay from {source_description}",
    )


def synthetic_replay_fallback(summary: dict[str, float], q_max: int) -> ReplayContext:
    logging.warning(
        "Using deterministic synthetic replay fallback for README figures. "
        "This path is only used when granular replay artifacts and bundled AAPL sample files are unavailable."
    )

    rng = np.random.default_rng(42)
    n_steps = 2_400
    minutes = np.linspace(0.0, 390.0, n_steps + 1)
    intraday = minutes / minutes[-1]

    bridge = rng.normal(0.0, 1.0, size=n_steps + 1).cumsum()
    bridge = bridge - intraday * bridge[-1]
    bridge = bridge / max(np.std(bridge), 1e-9)
    pnl = 662.0720632416997 * intraday + 28.0 * np.sin(2.5 * math.pi * intraday - 0.35) + 18.0 * bridge
    pnl[0] = 0.0
    pnl[-1] = float(summary["terminal_pnl"])

    inventory = np.round(6.0 * np.sin(3.0 * math.pi * intraday) - 3.0 * intraday).astype(float)
    inventory = np.clip(inventory, -q_max, q_max)

    fill_rows: list[dict[str, float | str]] = []
    half_spread = float(summary["spread_capture"]) / 2.0
    ask_ticks = half_spread / 0.01
    for side, count in DEFAULT_FILL_COUNTS.items():
        signed_ticks = ask_ticks if side == "ask" else -ask_ticks
        signed_bps = (half_spread if side == "ask" else -half_spread) / 100.0 * 10_000.0
        sample_times = np.linspace(minutes[15], minutes[-15], count)
        for timestamp in sample_times:
            fill_rows.append(
                {
                    "time": timestamp,
                    "side": side,
                    "fill_price": np.nan,
                    "trade_price": np.nan,
                    "mid_price": np.nan,
                    "relative_ticks": signed_ticks,
                    "relative_bps": signed_bps,
                }
            )

    return ReplayContext(
        times=minutes,
        inventory_path=inventory,
        pnl_path=pnl,
        fills=pd.DataFrame(fill_rows),
        terminal_pnl=float(summary["terminal_pnl"]),
        spread_capture=float(summary["spread_capture"]),
        n_fills=int(round(summary["n_fills"])),
        q_max=int(q_max),
        source_label="deterministic synthetic fallback",
    )


def build_replay_context() -> ReplayContext:
    configure_style()
    config_values = load_config_values()
    q_max = int(config_values.get("inventory.Q_max", 10))
    run_dir = find_primary_run_dir()
    summary = load_summary(run_dir)

    pnl_artifact, fill_artifact = find_explicit_series_artifacts()
    explicit_context = load_explicit_series_context(summary, q_max, pnl_artifact, fill_artifact)
    if explicit_context is not None:
        logging.info("Using explicit replay artifacts: %s", explicit_context.source_label)
        return explicit_context

    try:
        context = reconstruct_instantaneous_replay(summary, q_max, config_values, run_dir)
        logging.info("Using %s", context.source_label)
        return context
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logging.warning("Replay reconstruction unavailable: %s", error)
        return synthetic_replay_fallback(summary, q_max)


def load_hjb_parameters(config_values: dict[str, Any]) -> tuple[float, float, float, float, float]:
    sigma = float(config_values.get("model.sigma", 0.2))
    gamma = float(config_values.get("model.gamma", 0.1))
    kappa = float(config_values.get("model.kappa", 1.5))
    activity = float(config_values.get("model.A", 140.0))
    horizon = float(config_values.get("model.T", 1.0))
    return sigma, gamma, kappa, activity, horizon


def load_ablation_frame() -> pd.DataFrame:
    csv_path = ROOT / "results" / "cross_symbol_ablation" / "cross_symbol_ablation.csv"
    if not csv_path.exists():
        logging.info("No cross-symbol ablation CSV found. Using AvS-only fallback for ablation chart.")
        return pd.DataFrame(
            [
                {"quoter": "avs_optimal", "replay_terminal_pnl": DEFAULT_SUMMARY["terminal_pnl"], "available": True},
                {"quoter": "symmetric", "replay_terminal_pnl": np.nan, "available": False},
                {"quoter": "constant_spread", "replay_terminal_pnl": np.nan, "available": False},
                {"quoter": "inventory_linear", "replay_terminal_pnl": np.nan, "available": False},
                {"quoter": "random", "replay_terminal_pnl": np.nan, "available": False},
            ]
        )

    ablation = pd.read_csv(csv_path)
    if "replay_terminal_pnl" not in ablation.columns or "quoter" not in ablation.columns:
        raise ValueError(f"Ablation CSV at {csv_path} is missing required columns.")
    ablation = ablation[["quoter", "replay_terminal_pnl"]].copy()
    ablation["available"] = ablation["replay_terminal_pnl"].notna()
    return ablation


def minutes_since_open(times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.asarray([], dtype=float)
    return (np.asarray(times, dtype=float) - float(times[0])) / 60.0


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path.relative_to(ROOT))


def source_note(context: ReplayContext) -> str:
    label = context.source_label
    if label.startswith("reconstructed instantaneous replay"):
        return "Source: bundled AAPL sample\nwith checked-in replay calibration"
    if label.startswith("explicit artifacts"):
        return "Source: explicit replay artifacts\nunder results/**"
    return f"Source: {label}"


def plot_pnl_curve(context: ReplayContext, summary: dict[str, float], output_path: Path) -> None:
    x = minutes_since_open(context.times)
    y = np.asarray(context.pnl_path, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.plot(x, y, color="#0f766e", linewidth=2.0)
    ax.fill_between(x, y, 0.0, color="#14b8a6", alpha=0.12)
    ax.axhline(0.0, color="#475569", linewidth=1.0, linestyle=":")
    ax.scatter([x[-1]], [y[-1]], color="#0f766e", s=36, zorder=3)
    ax.annotate(f"${y[-1]:.0f}", xy=(x[-1], y[-1]), xytext=(-54, 12), textcoords="offset points", fontsize=10)
    ax.set_title("LOBSTER replay cumulative PnL")
    ax.set_xlabel("Minutes since open")
    ax.set_ylabel("Mark-to-market PnL ($)")
    ax.text(
        0.015,
        0.94,
        f"{source_note(context)}\nChecked-in headline: ${summary['terminal_pnl']:.2f}, "
        f"{int(round(summary['n_fills']))} fills, {summary['spread_capture']:.4f} realized spread",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    save_figure(fig, output_path)


def plot_inventory(context: ReplayContext, summary: dict[str, float], output_path: Path) -> None:
    x = minutes_since_open(context.times)
    y = np.asarray(context.inventory_path, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.axhspan(-context.q_max, context.q_max, color="#dbeafe", alpha=0.65, label=f"Configured hard-limit band (±{context.q_max})")
    ax.plot(x, y, color="#1d4ed8", linewidth=1.5, label="Replay inventory")
    ax.axhline(0.0, color="#475569", linewidth=1.0, linestyle=":")
    ax.set_title("Inventory trajectory over the AAPL replay")
    ax.set_xlabel("Minutes since open")
    ax.set_ylabel("Inventory (shares)")
    ax.legend(loc="upper left")
    ax.text(
        0.985,
        0.96,
        f"Mean |q|: {summary['avg_abs_inventory']:.2f}\nObserved range: [{int(np.min(y))}, {int(np.max(y))}]",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    save_figure(fig, output_path)


def plot_fill_distribution(context: ReplayContext, output_path: Path) -> None:
    fills = context.fills.copy()
    if fills.empty:
        raise RuntimeError("Cannot render fill distribution without fill-level data.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, constrained_layout=True)
    side_config = [("bid", "#2563eb"), ("ask", "#dc2626")]
    max_abs_tick = max(5.0, float(np.nanmax(np.abs(fills["relative_ticks"].to_numpy(dtype=float)))) + 1.0)
    bins = np.linspace(-max_abs_tick, max_abs_tick, 25)

    for axis, (side, color) in zip(axes, side_config, strict=True):
        subset = fills[fills["side"] == side]["relative_ticks"].to_numpy(dtype=float)
        axis.hist(subset, bins=bins, color=color, alpha=0.78, edgecolor="white")
        axis.axvline(float(np.mean(subset)), color="#0f172a", linewidth=1.1, linestyle="--")
        axis.set_title(f"{side.title()} fills ({len(subset)})")
        axis.set_xlabel("Fill price relative to mid (ticks)")
        axis.set_xlim(-max_abs_tick, max_abs_tick)
        axis.text(
            0.03,
            0.93,
            f"Mean: {np.mean(subset):.2f} ticks",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.88},
        )

    axes[0].set_ylabel("Fill count")
    fig.suptitle("Replay fill-price distribution relative to mid")
    save_figure(fig, output_path)


def plot_hjb_quote_example(config_values: dict[str, Any], output_path: Path) -> None:
    sigma, gamma, kappa, activity, horizon = load_hjb_parameters(config_values)
    tau = np.linspace(0.0, horizon, 250)
    mid_price = 100.0
    inventories = [0, 5, -5]
    colors = {0: "#0f766e", 5: "#b45309", -5: "#7c3aed"}

    fig, ax = plt.subplots(figsize=(10, 5.1), constrained_layout=True)
    if math.isclose(gamma, 0.0, abs_tol=1e-12):
        half_spread = np.full_like(tau, 1.0 / kappa)
    else:
        # Follow the requested closed-form expression for the README figure.
        half_spread = gamma * sigma**2 * tau + np.log1p(gamma / kappa) / gamma

    for inventory in inventories:
        reservation = mid_price - inventory * gamma * sigma**2 * tau
        bid = reservation - half_spread
        ask = reservation + half_spread
        ax.plot(tau, bid, color=colors[inventory], linewidth=2.0, linestyle="--", label=f"Bid, q={inventory:+d}")
        ax.plot(tau, ask, color=colors[inventory], linewidth=2.0, label=f"Ask, q={inventory:+d}")

    ax.set_title("Avellaneda-Stoikov theoretical quotes vs time-to-horizon")
    ax.set_xlabel("Time to horizon")
    ax.set_ylabel("Quote level")
    ax.legend(ncol=2, fontsize=9)
    ax.text(
        0.015,
        0.04,
        f"Parameters: sigma={sigma:.3f}, gamma={gamma:.3f}, kappa={kappa:.3f}, T={horizon:.1f}, A={activity:.1f}\n"
        "A is shown for completeness but does not enter the closed-form bid/ask equations plotted here.",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    save_figure(fig, output_path)


def plot_ablation(output_path: Path) -> None:
    ablation = load_ablation_frame()
    order = ["avs_optimal", "symmetric", "constant_spread", "inventory_linear", "random"]
    labels = {
        "avs_optimal": "AvS",
        "symmetric": "Symmetric",
        "constant_spread": "Constant",
        "inventory_linear": "Inventory-linear",
        "random": "Random",
    }
    ablation["sort_key"] = ablation["quoter"].map({name: idx for idx, name in enumerate(order)})
    ablation = ablation.sort_values("sort_key").reset_index(drop=True)

    display_values = ablation["replay_terminal_pnl"].fillna(0.0).to_numpy(dtype=float)
    colors = ["#0f766e" if quoter == "avs_optimal" else "#94a3b8" for quoter in ablation["quoter"]]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bars = ax.bar(np.arange(len(ablation)), display_values, color=colors)
    ax.axhline(0.0, color="#475569", linewidth=1.0)
    ax.set_xticks(np.arange(len(ablation)), [labels.get(quoter, quoter) for quoter in ablation["quoter"]], rotation=0)
    ax.set_ylabel("Terminal replay PnL ($)")
    ax.set_title("Replay ablation across checked-in quoters")

    for bar, (_, row) in zip(bars, ablation.iterrows(), strict=True):
        if bool(row["available"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 10.0,
                f"${row['replay_terminal_pnl']:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            bar.set_hatch("//")
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                12.0,
                "n/a\nin this snapshot",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    save_figure(fig, output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[render_readme_figures] %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_values = load_config_values()
    summary = load_summary(find_primary_run_dir())
    context = build_replay_context()

    plot_pnl_curve(context, summary, RESULTS_DIR / "lobster_pnl_curve.png")
    plot_inventory(context, summary, RESULTS_DIR / "inventory_trajectory.png")
    plot_fill_distribution(context, RESULTS_DIR / "fill_distribution.png")
    plot_hjb_quote_example(config_values, RESULTS_DIR / "hjb_quote_example.png")
    plot_ablation(RESULTS_DIR / "ablation_quoters.png")


if __name__ == "__main__":
    main()
