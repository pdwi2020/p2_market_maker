"""CLI entrypoint for the Week 2 cross-symbol LOBSTER ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from p2.baselines import AvSOptimalMM, ConstantSpreadMM, InventoryLinearMM, RandomQuoter, SymmetricMM
from p2.cross_symbol_sweep import (
    DEFAULT_DATE,
    DEFAULT_REPLAY_GAMMA,
    DEFAULT_REPLAY_T,
    LOBSTER_FREE_SAMPLE_URLS,
    run_symbol_sweep,
)
from p2.config import load_config
from p2.hjb_solver import optimal_spread


def _ordered_symbols(requested: list[str] | None) -> list[str]:
    universe = list(LOBSTER_FREE_SAMPLE_URLS.keys())
    if not requested:
        return universe
    requested_upper = [symbol.upper() for symbol in requested]
    remainder = [symbol for symbol in universe if symbol not in requested_upper]
    return requested_upper + remainder


def _default_quoters() -> dict[str, object]:
    def half_spread(calibration: dict[str, float | int | str]) -> float:
        return optimal_spread(
            t=0.0,
            T=DEFAULT_REPLAY_T,
            gamma=DEFAULT_REPLAY_GAMMA,
            sigma=float(calibration["sigma"]),
            kappa=float(calibration["kappa"]),
        )

    return {
        "avs_optimal": lambda calibration: AvSOptimalMM(
            sigma=float(calibration["sigma"]),
            gamma=DEFAULT_REPLAY_GAMMA,
            kappa=float(calibration["kappa"]),
            T=DEFAULT_REPLAY_T,
        ),
        "symmetric": lambda calibration: SymmetricMM(half_spread=half_spread(calibration)),
        "constant_spread": lambda calibration: ConstantSpreadMM(
            sigma=float(calibration["sigma"]),
            gamma=DEFAULT_REPLAY_GAMMA,
            kappa=float(calibration["kappa"]),
            T=DEFAULT_REPLAY_T,
        ),
        "inventory_linear": lambda calibration: InventoryLinearMM(
            half_spread=half_spread(calibration),
            lambda_q=1.0,
            Q_max=10,
        ),
        "random": lambda calibration: RandomQuoter(
            half_spread_max=half_spread(calibration),
            rng_seed=0,
        ),
    }


def main() -> None:
    default_lobster_dir = load_config().paths.lobster_dir
    parser = argparse.ArgumentParser(description="Run the cross-symbol LOBSTER ablation sweep.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--data-dir", type=Path, default=default_lobster_dir)
    parser.add_argument("--results-dir", default="results/cross_symbol_ablation")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    symbols = _ordered_symbols(args.symbols)
    run_symbol_sweep(
        symbols=symbols,
        date=args.date,
        data_dir=args.data_dir,
        results_dir=results_dir,
        quoters=_default_quoters(),
    )
    summary = json.loads((results_dir / "summary.json").read_text())
    print(f"saved_results={results_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
