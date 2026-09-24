"""Published table and figure generation for the market-making study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


TABLE_FILES = {
    "daily_pnl": "daily_pnl.csv",
    "decomposition": "decomposition.csv",
    "markouts": "markouts.csv",
    "queue_validation": "queue_validation.csv",
    "hftbacktest_crosscheck": "hftbacktest_crosscheck.csv",
    "lobster_appendix": "lobster_appendix.csv",
}

METHOD_ORDER = ("exact_fifo", "l2_cancel_from_back", "l2_proportional")

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


def _short_label(strategy: str) -> str:
    """Wrap a strategy name so bar charts do not turn into vertical text."""
    parts = strategy.split("_")
    if parts[0] == "glft" and len(parts) > 1 and parts[1] == "imbalance":
        head, tail = "glft imbalance", parts[2:]
    else:
        head, tail = parts[0], parts[1:]
    if not tail:
        return head
    return head + "\n" + " ".join(tail).replace("gamma ", "gamma=").replace("beta ", "beta=")


def _apply_strategy_ticks(axis: plt.Axes, strategies: list[str]) -> None:
    axis.set_xticks(range(len(strategies)))
    axis.set_xticklabels([_short_label(name) for name in strategies], fontsize=8)
    axis.tick_params(axis="x", rotation=0)


def _scale_for(values: np.ndarray) -> tuple[str, float]:
    """Pick a y-scale that keeps small series visible next to large ones."""
    magnitudes = np.abs(values[np.isfinite(values) & (values != 0.0)])
    if magnitudes.size < 2:
        return "linear", 0.0
    spread = magnitudes.max() / magnitudes.min()
    if spread <= 20.0:
        return "linear", 0.0
    return "symlog", float(magnitudes.min())


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
        finals = []
        # Two strategies can end within a fraction of a per cent of each other,
        # in which case one line hides the other completely. Alternating dash
        # patterns keep every series in the legend visible on the canvas.
        styles = ("-", "--", "-.", ":")
        for index, (strategy, rows) in enumerate(primary.groupby("strategy", sort=True)):
            rows = rows.sort_values("date")
            dates = pd.to_datetime(rows["date"])
            cumulative = rows["net_pnl"].cumsum()
            finals.append(cumulative.iloc[-1])
            axis.plot(
                dates,
                cumulative,
                label=strategy,
                linestyle=styles[index % len(styles)],
                linewidth=1.6,
            )
        axis.legend(frameon=False, fontsize=8)
        # One strategy can end two orders of magnitude from another, which on a
        # linear axis flattens it onto zero and hides its shape entirely.
        scale, linthresh = _scale_for(np.asarray(finals, dtype=float))
        if scale == "symlog":
            axis.set_yscale("symlog", linthresh=linthresh)
            axis.set_ylabel("USDT (symmetric log scale)")
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axis.xaxis.set_minor_locator(mdates.MonthLocator())
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Cumulative BTCUSDT net PnL")
    axis.set_xlabel("UTC date")
    if not axis.get_ylabel():
        axis.set_ylabel("USDT")
    axis.tick_params(axis="x", rotation=30)
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
        # fee_cost is published as a positive magnitude; the identity is
        # realized spread plus inventory revaluation minus fees.
        totals["fee_cost"] *= -1.0
        totals.plot(kind="bar", ax=axis, rot=0)
        axis.legend(frameon=False, fontsize=8)
        _apply_strategy_ticks(axis, list(totals.index))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("BTCUSDT PnL decomposition")
    axis.set_xlabel("")
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
        axis.set_xticks(range(1, len(strategies) + 1))
        axis.set_xticklabels(
            [_short_label(name) for name in strategies], fontsize=8
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("BTCUSDT end-of-day inventory")
    axis.set_xlabel("")
    axis.set_ylabel("BTC")
    _save(fig, destination / "inventory_distribution.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    aggregate_queue = queue_validation.copy()
    if "date" in aggregate_queue and (aggregate_queue["date"] == "ALL").any():
        aggregate_queue = aggregate_queue[aggregate_queue["date"] == "ALL"]
    metrics = (
        ("fill_count_relative_bias", "Fill-count bias", "fraction of exact fills"),
        ("fill_time_ks", "Fill-time KS", "KS statistic"),
        ("gross_pnl_error", "Gross-PnL error", "USD against exact FIFO"),
    )
    # Each quoting arm is a separate comparison, so averaging them together
    # would hide exactly the differences the arms were added to show.
    arms = (
        sorted(aggregate_queue["arm"].unique())
        if "arm" in aggregate_queue and not aggregate_queue.empty
        else []
    )
    present = (
        set(aggregate_queue["method"]) if not aggregate_queue.empty else set()
    )
    methods = [name for name in METHOD_ORDER if name in present]
    width = 0.8 / max(len(arms), 1)
    for axis, (column, title, units) in zip(axes, metrics, strict=True):
        for index, arm in enumerate(arms):
            scoped = aggregate_queue[aggregate_queue["arm"] == arm]
            values = [
                float(scoped.loc[scoped["method"] == method, column].mean())
                if (scoped["method"] == method).any()
                else np.nan
                for method in methods
            ]
            positions = np.arange(len(methods)) + index * width - 0.4 + width / 2
            axis.bar(positions, values, width=width, label=arm)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel(units, fontsize=8)
        axis.set_xticks(range(len(methods)))
        axis.set_xticklabels(
            [name.replace("l2_", "L2 ").replace("_", " ") for name in methods],
            fontsize=8,
            rotation=20,
            ha="right",
        )
    if arms:
        axes[0].legend(frameon=False, fontsize=7, title="arm", title_fontsize=7)
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
