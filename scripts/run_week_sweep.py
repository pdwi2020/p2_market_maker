"""Week 2 five-scenario regime-stressed LOBSTER-style sweep.

Run with the shared venv:
    /Volumes/Crucial X9/alpha_engine/.venv/bin/python scripts/run_week_sweep.py

Deterministic seeds:
    scenario_0_base: 20260424
    scenario_1_high_vol: 20260425
    scenario_2_low_vol: 20260426
    scenario_3_thin_book: 20260427
    scenario_4_adverse_selection: 20260428

The checked-in AAPL replay calibration does not store gamma. This script
re-selects a replay-consistent gamma from a small deterministic grid so the
base scenario stays anchored to the existing `$662 / 972 fills / 7.9%`
artifact before applying the synthetic perturbations.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / "results" / "lobster_aapl" / "calibration.json"
BASELINE_PATH = ROOT / "results" / "lobster_aapl" / "replay_summary.json"
OUTPUT_ROOT = ROOT / "results" / "lobster_week_sweep"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from p2.adverse_selection import ArrivalModel
from p2.glosten_milgrom import GlostenMilgromLayer, gm_adjusted_intensity
from p2.hjb_solver import optimal_ask, optimal_bid
from p2.lobster_replay import LOBSTERReplayer


BASE_PI = 0.10
STRESS_PI = 0.30
BASELINE_TOLERANCE = 0.15
GAMMA_GRID = [0.1, 0.05, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003]


@dataclass(frozen=True, slots=True)
class ReplayInputs:
    mids: np.ndarray
    prices: np.ndarray
    times: np.ndarray
    event_type: np.ndarray
    direction: np.ndarray
    n_rows: int
    start_time: float
    end_time: float
    horizon: float


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    idx: int
    name: str
    sigma: float
    A: float
    kappa: float
    gamma: float
    pi: float
    seed: int
    thin_book: bool = False
    adverse_selection: bool = False

    @property
    def label(self) -> str:
        return f"scenario_{self.idx}_{self.name}"


@dataclass(slots=True)
class ScenarioResult:
    spec: ScenarioSpec
    metrics: dict[str, float | int | str]
    pnl_series: pd.DataFrame
    inventory_series: pd.DataFrame


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_replay_inputs(calibration: dict[str, Any]) -> ReplayInputs:
    replayer = LOBSTERReplayer().load(calibration["orderbook_file"], calibration["message_file"])
    if replayer.orderbook is None or replayer.messages is None:
        raise RuntimeError("Failed to load LOBSTER replay inputs.")

    mids = replayer.mid_price_series().to_numpy(dtype=float)
    messages = replayer.messages
    prices = messages["price"].to_numpy(dtype=float)
    times = messages["time"].to_numpy(dtype=float)
    event_type = messages["event_type"].to_numpy(dtype=int)
    direction = messages["direction"].to_numpy(dtype=int)
    n_rows = min(len(mids), len(prices), len(times), len(event_type), len(direction))
    if n_rows == 0:
        raise RuntimeError("LOBSTER replay inputs are empty.")

    start_time = float(times[0])
    end_time = float(times[n_rows - 1])
    horizon = max(end_time - start_time, 1e-12)
    return ReplayInputs(
        mids=mids[:n_rows],
        prices=prices[:n_rows],
        times=times[:n_rows],
        event_type=event_type[:n_rows],
        direction=direction[:n_rows],
        n_rows=n_rows,
        start_time=start_time,
        end_time=end_time,
        horizon=horizon,
    )


def compute_max_drawdown(cum_pnl: np.ndarray) -> float:
    running_peak = np.maximum.accumulate(np.asarray(cum_pnl, dtype=float))
    drawdowns = running_peak - np.asarray(cum_pnl, dtype=float)
    return float(np.max(drawdowns)) if drawdowns.size else 0.0


def sharpe_ratio(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return 0.0
    std = float(np.std(array, ddof=1))
    return float(np.mean(array) / std) if std > 0.0 else 0.0


def minute_bucket_metrics(pnl_series: pd.DataFrame) -> tuple[float, float]:
    frame = pnl_series.copy()
    frame["minute_bucket"] = ((frame["time"] - frame["time"].iloc[0]) // 60.0).astype(int)
    minute = frame.groupby("minute_bucket", as_index=False).last()
    pnl_delta = minute["cum_pnl"].diff().fillna(minute["cum_pnl"])
    base_mid = minute["mid"].shift(1).fillna(minute["mid"])
    rel_returns = pnl_delta / base_mid.replace(0.0, np.nan)
    sharpe = sharpe_ratio(pnl_delta)
    info_ratio = sharpe_ratio(rel_returns.fillna(0.0))
    return sharpe, info_ratio


def baseline_score(
    metrics: dict[str, float | int | str],
    replay_anchor: dict[str, Any],
) -> float:
    pnl_target = float(replay_anchor["terminal_pnl"])
    fills_target = float(replay_anchor["n_fills"])
    spread_target = float(replay_anchor["spread_capture"])
    pnl_err = abs(float(metrics["terminal_pnl"]) - pnl_target) / max(abs(pnl_target), 1e-12)
    fills_err = abs(float(metrics["n_fills"]) - fills_target) / max(abs(fills_target), 1e-12)
    spread_err = abs(float(metrics["spread_capture"]) - spread_target) / max(abs(spread_target), 1e-12)
    return pnl_err + fills_err + spread_err


def run_replay_scenario(
    inputs: ReplayInputs,
    spec: ScenarioSpec,
    *,
    base_A: float,
    base_kappa: float,
    base_epsilon: float,
    base_pi: float,
) -> ScenarioResult:
    rng = np.random.default_rng(spec.seed)
    gm_layer = GlostenMilgromLayer(mu=spec.pi, v_high=1.0, v_low=-1.0, prior_high=0.5)

    inventory = 0
    cash = 0.0
    posterior = 0.5
    n_fills = 0
    realized_spreads: list[float] = []

    time_path = np.empty(inputs.n_rows + 1, dtype=float)
    inventory_path = np.zeros(inputs.n_rows + 1, dtype=float)
    pnl_path = np.zeros(inputs.n_rows + 1, dtype=float)
    mid_path = np.empty(inputs.n_rows + 1, dtype=float)

    time_path[0] = inputs.start_time
    mid_path[0] = float(inputs.mids[0])

    for row_idx in range(inputs.n_rows):
        mid = float(inputs.mids[row_idx])
        trade_price = float(inputs.prices[row_idx])
        event_type = int(inputs.event_type[row_idx])
        trade_sign = int(np.sign(inputs.direction[row_idx]))
        t_norm = row_idx / max(inputs.n_rows - 1, 1)

        bid = float(optimal_bid(mid, inventory, t_norm, 1.0, spec.gamma, spec.sigma, spec.kappa))
        ask = float(optimal_ask(mid, inventory, t_norm, 1.0, spec.gamma, spec.sigma, spec.kappa))

        if spec.adverse_selection:
            toxic_buy = max(posterior - 0.5, 0.0)
            toxic_sell = max(0.5 - posterior, 0.0)
            gm_skew = 4.0 * base_epsilon * spec.pi
            ask += gm_skew * toxic_buy
            bid -= gm_skew * toxic_sell

        if event_type in {4, 5} and trade_sign != 0:
            execute = False
            execute_prob = 1.0

            if trade_sign > 0 and ask <= trade_price:
                delta = max(ask - mid, 0.0)
                if spec.thin_book:
                    lambda_base = float(ArrivalModel.intensity(delta, A=base_A, kappa=base_kappa))
                    lambda_stress = float(ArrivalModel.intensity(delta, A=spec.A, kappa=spec.kappa))
                    execute_prob *= min(1.0, lambda_stress / max(lambda_base, 1e-12))
                if spec.adverse_selection:
                    lambda_base_buy, _ = gm_adjusted_intensity(
                        delta,
                        A=base_A,
                        kappa=base_kappa,
                        posterior=posterior,
                        mu=base_pi,
                    )
                    lambda_stress_buy, _ = gm_adjusted_intensity(
                        delta,
                        A=spec.A,
                        kappa=spec.kappa,
                        posterior=posterior,
                        mu=spec.pi,
                    )
                    execute_prob *= min(1.0, lambda_base_buy / max(lambda_stress_buy, 1e-12))
                execute = rng.random() <= execute_prob
                if execute:
                    cash += ask
                    inventory -= 1
                    n_fills += 1
                    realized_spreads.append(ask - bid)
            elif trade_sign < 0 and bid >= trade_price:
                delta = max(mid - bid, 0.0)
                if spec.thin_book:
                    lambda_base = float(ArrivalModel.intensity(delta, A=base_A, kappa=base_kappa))
                    lambda_stress = float(ArrivalModel.intensity(delta, A=spec.A, kappa=spec.kappa))
                    execute_prob *= min(1.0, lambda_stress / max(lambda_base, 1e-12))
                if spec.adverse_selection:
                    _, lambda_base_sell = gm_adjusted_intensity(
                        delta,
                        A=base_A,
                        kappa=base_kappa,
                        posterior=posterior,
                        mu=base_pi,
                    )
                    _, lambda_stress_sell = gm_adjusted_intensity(
                        delta,
                        A=spec.A,
                        kappa=spec.kappa,
                        posterior=posterior,
                        mu=spec.pi,
                    )
                    execute_prob *= min(1.0, lambda_base_sell / max(lambda_stress_sell, 1e-12))
                execute = rng.random() <= execute_prob
                if execute:
                    cash -= bid
                    inventory += 1
                    n_fills += 1
                    realized_spreads.append(ask - bid)

            posterior = gm_layer.posterior(trade_sign, prior=posterior)

        time_path[row_idx + 1] = float(inputs.times[row_idx])
        mid_path[row_idx + 1] = mid
        inventory_path[row_idx + 1] = float(inventory)
        pnl_path[row_idx + 1] = cash + inventory * mid

    pnl_series = pd.DataFrame({"time": time_path, "cum_pnl": pnl_path, "mid": mid_path})
    inventory_series = pd.DataFrame({"time": time_path, "inventory": inventory_path.astype(int)})
    sharpe, info_ratio = minute_bucket_metrics(pnl_series)

    metrics = {
        "scenario": spec.label,
        "scenario_name": spec.name,
        "seed": spec.seed,
        "sigma": float(spec.sigma),
        "A": float(spec.A),
        "kappa": float(spec.kappa),
        "gamma": float(spec.gamma),
        "pi": float(spec.pi),
        "terminal_pnl": float(pnl_path[-1]),
        "inventory_variance": float(np.var(inventory_path)),
        "avg_abs_inventory": float(np.mean(np.abs(inventory_path))),
        "spread_capture": float(np.mean(realized_spreads)) if realized_spreads else 0.0,
        "n_fills": int(n_fills),
        "max_drawdown": compute_max_drawdown(pnl_path),
        "sharpe": float(sharpe),
        "information_ratio": float(info_ratio),
    }
    return ScenarioResult(
        spec=spec,
        metrics=metrics,
        pnl_series=pnl_series[["time", "cum_pnl"]],
        inventory_series=inventory_series,
    )


def select_gamma(inputs: ReplayInputs, calibration: dict[str, Any], replay_anchor: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    best_gamma = float(GAMMA_GRID[0])
    best_result: ScenarioResult | None = None
    best_score = math.inf

    for gamma in GAMMA_GRID:
        spec = ScenarioSpec(
            idx=0,
            name="base",
            sigma=float(calibration["sigma"]),
            A=float(calibration["A"]),
            kappa=float(calibration["kappa"]),
            gamma=float(gamma),
            pi=BASE_PI,
            seed=20260424,
        )
        result = run_replay_scenario(
            inputs,
            spec,
            base_A=float(calibration["A"]),
            base_kappa=float(calibration["kappa"]),
            base_epsilon=float(calibration["epsilon"]),
            base_pi=BASE_PI,
        )
        score = baseline_score(result.metrics, replay_anchor)
        if score < best_score:
            best_score = score
            best_gamma = float(gamma)
            best_result = result

    if best_result is None:
        raise RuntimeError("Failed to select a replay-consistent gamma.")
    return best_gamma, best_result.metrics


def scenario_specs(calibration: dict[str, Any], gamma: float) -> list[ScenarioSpec]:
    sigma = float(calibration["sigma"])
    activity = float(calibration["A"])
    kappa = float(calibration["kappa"])
    return [
        ScenarioSpec(0, "base", sigma, activity, kappa, gamma, BASE_PI, 20260424),
        ScenarioSpec(1, "high_vol", 2.0 * sigma, activity, kappa, gamma, BASE_PI, 20260425),
        ScenarioSpec(2, "low_vol", 0.5 * sigma, activity, kappa, gamma, BASE_PI, 20260426),
        ScenarioSpec(3, "thin_book", sigma, 0.5 * activity, 2.0 * kappa, gamma, BASE_PI, 20260427, thin_book=True),
        ScenarioSpec(4, "adverse_selection", sigma, activity, kappa, gamma, STRESS_PI, 20260428, adverse_selection=True),
    ]


def save_scenario_artifacts(result: ScenarioResult) -> None:
    scenario_dir = OUTPUT_ROOT / result.spec.label
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    result.pnl_series.to_csv(scenario_dir / "pnl_series.csv", index=False)
    result.inventory_series.to_csv(scenario_dir / "inventory_series.csv", index=False)


def write_summary(
    results: list[ScenarioResult],
    gamma: float,
    replay_anchor: dict[str, Any],
) -> pd.DataFrame:
    summary_frame = pd.DataFrame([result.metrics for result in results])
    aggregate = {
        "mean_pnl": float(summary_frame["terminal_pnl"].mean()),
        "std_pnl": float(summary_frame["terminal_pnl"].std(ddof=1)) if len(summary_frame) > 1 else 0.0,
        "min_drawdown": float(summary_frame["max_drawdown"].min()),
        "mean_sharpe": float(summary_frame["sharpe"].mean()),
        "mean_ir": float(summary_frame["information_ratio"].mean()),
    }
    summary_payload = {
        "gamma_selected": float(gamma),
        "baseline_tolerance": BASELINE_TOLERANCE,
        "replay_anchor": replay_anchor,
        "scenarios": summary_frame.to_dict(orient="records"),
        "aggregate": aggregate,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    summary_frame.to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    return summary_frame


def plot_pnl_curves(results: list[ScenarioResult]) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {
        "scenario_0_base": "#0b5fff",
        "scenario_1_high_vol": "#4f6fad",
        "scenario_2_low_vol": "#8fb8ff",
        "scenario_3_thin_book": "#9a6b2f",
        "scenario_4_adverse_selection": "#b42318",
    }

    for result in results:
        label = result.spec.label
        series = result.pnl_series
        linewidth = 2.8 if result.spec.idx == 0 else 1.4
        ax.plot(
            series["time"],
            series["cum_pnl"],
            label=label,
            color=colors.get(label, "#5f6b7a"),
            linewidth=linewidth,
            alpha=0.95 if result.spec.idx == 0 else 0.9,
        )
        terminal_pnl = float(series["cum_pnl"].iloc[-1])
        ax.annotate(
            f"{terminal_pnl:.1f}",
            xy=(float(series["time"].iloc[-1]), terminal_pnl),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color=colors.get(label, "#5f6b7a"),
            va="center",
        )

    ax.set_title("Week 2 Regime-Stressed Cumulative PnL")
    ax.set_xlabel("LOBSTER event time (seconds)")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "pnl_curves.png", dpi=180)
    plt.close(fig)


def plot_metric_bars(summary_frame: pd.DataFrame) -> None:
    metrics = [
        ("terminal_pnl", "Terminal PnL"),
        ("max_drawdown", "Max Drawdown"),
        ("sharpe", "Sharpe"),
        ("information_ratio", "Information Ratio"),
    ]
    labels = summary_frame["scenario"].tolist()
    colors = ["#0b5fff" if label == "scenario_0_base" else "#b7bec8" for label in labels]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, title) in zip(axes.flatten(), metrics):
        ax.bar(labels, summary_frame[column], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "metrics_bars.png", dpi=180)
    plt.close(fig)


def print_baseline_check(base_metrics: dict[str, Any], replay_anchor: dict[str, Any]) -> bool:
    pnl_target = float(replay_anchor["terminal_pnl"])
    fills_target = float(replay_anchor["n_fills"])
    spread_target = float(replay_anchor["spread_capture"])

    pnl_ok = abs(float(base_metrics["terminal_pnl"]) - pnl_target) <= BASELINE_TOLERANCE * abs(pnl_target)
    fills_ok = abs(float(base_metrics["n_fills"]) - fills_target) <= BASELINE_TOLERANCE * abs(fills_target)
    spread_ok = abs(float(base_metrics["spread_capture"]) - spread_target) <= BASELINE_TOLERANCE * abs(spread_target)

    print("Baseline tolerance check (target +/-15% from results/lobster_aapl/replay_summary.json)")
    print(
        "target",
        json.dumps(
            {
                "terminal_pnl": pnl_target,
                "n_fills": fills_target,
                "spread_capture": spread_target,
            }
        ),
    )
    print(
        "scenario_0_actual",
        json.dumps(
            {
                "terminal_pnl": float(base_metrics["terminal_pnl"]),
                "n_fills": int(base_metrics["n_fills"]),
                "spread_capture": float(base_metrics["spread_capture"]),
            }
        ),
    )
    print(f"baseline_match_pnl={str(pnl_ok).lower()}")
    print(f"baseline_match_fills={str(fills_ok).lower()}")
    print(f"baseline_match_spread_capture={str(spread_ok).lower()}")
    return pnl_ok


def main() -> None:
    calibration = _load_json(CALIBRATION_PATH)
    replay_anchor = _load_json(BASELINE_PATH)
    inputs = load_replay_inputs(calibration)

    gamma, gamma_anchor_metrics = select_gamma(inputs, calibration, replay_anchor)
    specs = scenario_specs(calibration, gamma)
    results = [
        run_replay_scenario(
            inputs,
            spec,
            base_A=float(calibration["A"]),
            base_kappa=float(calibration["kappa"]),
            base_epsilon=float(calibration["epsilon"]),
            base_pi=BASE_PI,
        )
        for spec in specs
    ]

    for result in results:
        save_scenario_artifacts(result)

    summary_frame = write_summary(results, gamma, replay_anchor)
    plot_pnl_curves(results)
    plot_metric_bars(summary_frame)

    print(f"selected_gamma={gamma:.6f}")
    print("gamma_anchor_metrics", json.dumps(gamma_anchor_metrics))
    base_within_pnl = print_baseline_check(results[0].metrics, replay_anchor)
    print("saved_results", OUTPUT_ROOT)
    print(
        "aggregate",
        json.dumps(
            {
                "mean_pnl": float(summary_frame["terminal_pnl"].mean()),
                "mean_sharpe": float(summary_frame["sharpe"].mean()),
                "mean_ir": float(summary_frame["information_ratio"].mean()),
                "scenario_0_pnl_within_baseline": "yes" if base_within_pnl else "no",
            }
        ),
    )


if __name__ == "__main__":
    main()
