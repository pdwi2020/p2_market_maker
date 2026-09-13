"""Published table and figure generation for the market-making study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TABLE_FILES = {
    "daily_pnl": "daily_pnl.csv",
    "decomposition": "decomposition.csv",
    "markouts": "markouts.csv",
    "queue_validation": "queue_validation.csv",
    "hftbacktest_crosscheck": "hftbacktest_crosscheck.csv",
    "lobster_appendix": "lobster_appendix.csv",
}

FIGURE_FILES = (
    "cumulative_net_pnl.png",
    "markout_curves.png",
    "pnl_decomposition.png",
    "inventory_distribution.png",
    "queue_model_error.png",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def write_research_tables(
    output_dir: str | Path,
    *,
    summary: dict[str, Any],
    daily_pnl: pd.DataFrame,
    decomposition: pd.DataFrame,
    markouts: pd.DataFrame,
    queue_validation: pd.DataFrame,
    hftbacktest_crosscheck: pd.DataFrame,
    lobster_appendix: pd.DataFrame,
) -> None:
    """Write the fixed published table contract."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frames = {
        "daily_pnl": daily_pnl,
        "decomposition": decomposition,
        "markouts": markouts,
        "queue_validation": queue_validation,
        "hftbacktest_crosscheck": hftbacktest_crosscheck,
        "lobster_appendix": lobster_appendix,
    }
    for key, filename in TABLE_FILES.items():
        frames[key].to_csv(destination / filename, index=False)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _primary_crypto(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    selected = daily.copy()
    if "symbol" in selected:
        selected = selected[selected["symbol"] == "BTCUSDT"]
    if "maker_fee_rate" in selected:
        selected = selected[np.isclose(selected["maker_fee_rate"], 0.0002)]
    if "latency_ms" in selected:
        selected = selected[selected["latency_ms"] == 50]
    return selected


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_research_figures(output_dir: str | Path) -> None:
    """Render all figures strictly from the published CSV tables."""
    destination = Path(output_dir)
    daily = pd.read_csv(destination / TABLE_FILES["daily_pnl"])
    decomposition = pd.read_csv(destination / TABLE_FILES["decomposition"])
    markouts = pd.read_csv(destination / TABLE_FILES["markouts"])
    queue_validation = pd.read_csv(destination / TABLE_FILES["queue_validation"])
    primary = _primary_crypto(daily)
    _style()

    fig, axis = plt.subplots(figsize=(8, 4.5))
    if not primary.empty:
        for strategy, rows in primary.groupby("strategy", sort=True):
            rows = rows.sort_values("date")
            axis.plot(rows["date"], rows["net_pnl"].cumsum(), label=strategy)
        axis.legend(frameon=False)
    axis.set_title("Cumulative BTCUSDT net PnL")
    axis.set_xlabel("UTC date")
    axis.set_ylabel("USDT")
    axis.tick_params(axis="x", rotation=35)
    _save(fig, destination / "cumulative_net_pnl.png")

    fig, axis = plt.subplots(figsize=(7, 4.5))
    primary_markouts = markouts.copy()
    if not primary_markouts.empty:
        if "symbol" in primary_markouts:
            primary_markouts = primary_markouts[
                primary_markouts["symbol"] == "BTCUSDT"
            ]
        if "maker_fee_rate" in primary_markouts:
            primary_markouts = primary_markouts[
                np.isclose(primary_markouts["maker_fee_rate"], 0.0002)
            ]
        if "latency_ms" in primary_markouts:
            primary_markouts = primary_markouts[primary_markouts["latency_ms"] == 50]
        for strategy, rows in primary_markouts.groupby("strategy", sort=True):
            grouped = rows.groupby("horizon_seconds")["mean_markout"].mean()
            axis.plot(grouped.index, grouped.values, marker="o", label=strategy)
        axis.legend(frameon=False)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Signed BTCUSDT markouts")
    axis.set_xlabel("Horizon (seconds)")
    axis.set_ylabel("USDT per fill")
    _save(fig, destination / "markout_curves.png")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    primary_decomposition = decomposition.copy()
    if not primary_decomposition.empty:
        if "symbol" in primary_decomposition:
            primary_decomposition = primary_decomposition[
                primary_decomposition["symbol"] == "BTCUSDT"
            ]
        if "maker_fee_rate" in primary_decomposition:
            primary_decomposition = primary_decomposition[
                np.isclose(primary_decomposition["maker_fee_rate"], 0.0002)
            ]
        if "latency_ms" in primary_decomposition:
            primary_decomposition = primary_decomposition[
                primary_decomposition["latency_ms"] == 50
            ]
        totals = primary_decomposition.groupby("strategy")[[
            "realized_spread",
            "inventory_revaluation",
            "fee_cost",
        ]].sum()
        totals["fee_cost"] *= -1.0
        totals.plot(kind="bar", ax=axis)
        axis.legend(frameon=False)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("BTCUSDT PnL decomposition")
    axis.set_xlabel("Strategy")
    axis.set_ylabel("USDT")
    _save(fig, destination / "pnl_decomposition.png")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    if not primary.empty:
        strategies = sorted(primary["strategy"].unique())
        distributions = [
            primary.loc[primary["strategy"] == strategy, "end_inventory"].to_numpy()
            for strategy in strategies
        ]
        axis.boxplot(distributions, tick_labels=strategies)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("BTCUSDT end-of-day inventory")
    axis.set_xlabel("Strategy")
    axis.set_ylabel("BTC")
    axis.tick_params(axis="x", rotation=25)
    _save(fig, destination / "inventory_distribution.png")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    aggregate_queue = queue_validation.copy()
    if "date" in aggregate_queue and (aggregate_queue["date"] == "ALL").any():
        aggregate_queue = aggregate_queue[aggregate_queue["date"] == "ALL"]
    metrics = (
        ("fill_count_relative_bias", "Fill-count bias"),
        ("fill_time_ks", "Fill-time KS"),
        ("gross_pnl_error", "Gross-PnL error"),
    )
    for axis, (column, title) in zip(axes, metrics, strict=True):
        if not aggregate_queue.empty and column in aggregate_queue:
            grouped = aggregate_queue.groupby("method")[column].mean().sort_index()
            axis.bar(grouped.index, grouped.values)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
    _save(fig, destination / "queue_model_error.png")


def publish_research_outputs(
    output_dir: str | Path,
    *,
    summary: dict[str, Any],
    daily_pnl: pd.DataFrame,
    decomposition: pd.DataFrame,
    markouts: pd.DataFrame,
    queue_validation: pd.DataFrame,
    hftbacktest_crosscheck: pd.DataFrame,
    lobster_appendix: pd.DataFrame,
) -> None:
    """Write all tables, then render all figures from those tables."""
    write_research_tables(
        output_dir,
        summary=summary,
        daily_pnl=daily_pnl,
        decomposition=decomposition,
        markouts=markouts,
        queue_validation=queue_validation,
        hftbacktest_crosscheck=hftbacktest_crosscheck,
        lobster_appendix=lobster_appendix,
    )
    render_research_figures(output_dir)
