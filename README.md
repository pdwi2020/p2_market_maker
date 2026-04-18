# P2 — Avellaneda-Stoikov Market Maker

## Overview
P2 implements the Avellaneda-Stoikov (2008) market-making model as a reproducible research repo rather than a toy simulator. The repo now has:

- a typed shared configuration layer in `configs/p2_config.yaml`
- a single synthetic execution engine used by AVS and both baseline strategies
- inventory hard limits enforced by quote suppression, not post-fill clipping
- configurable adverse selection via immediate post-fill mid-price jumps
- a parameter sweep that uses the same fill, inventory, and adverse-selection semantics as the default simulator
- a simplified top-of-book LOBSTER replay and calibration path that works when sample CSVs are available

The canonical environment is `/Volumes/Crucial X9/alpha_engine/.venv`. No repo-local `.venv` is used.

## Model
The mid price follows arithmetic Brownian motion

\[
dS_t = \sigma dW_t
\]

with Poisson arrivals

\[
\lambda(\delta) = A e^{-\kappa \delta}.
\]

The reservation price and total spread are

\[
r(q,t) = S_t - q \gamma \sigma^2 (T-t),
\]

\[
\Delta^*(t) = \gamma \sigma^2 (T-t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right),
\]

and the code places quotes using the one-sided half-spread

\[
b_t = r_t - \frac{\Delta^*(t)}{2}, \qquad a_t = r_t + \frac{\Delta^*(t)}{2}.
\]

## Setup
```bash
make install
```

This installs P2 into the shared drive environment with `pip install -e .[dev]`.

## Commands
```bash
make simulate
make sweep
make calibrate
make backtest
make test
```

The default config is `configs/p2_config.yaml`. Replay commands fail with a clear message when `data/lobster/` is empty.

## Current Default Results
These values come from `make simulate` and `make sweep` using `configs/p2_config.yaml` on the current machine.

### Default synthetic run
Artifact directory: `results/default_synthetic/`

| Experiment | Mean terminal PnL | PnL std | Sharpe | Avg abs inventory | Spread capture |
| --- | ---: | ---: | ---: | ---: | ---: |
| AVS | 64.5116 | 6.6831 | 9.6530 | 2.0828 | 67.7794 |
| Symmetric | 60.1049 | 7.7000 | 7.8058 | 4.4297 | 61.7990 |
| Constant spread | 63.4168 | 8.5089 | 7.4530 | 4.2453 | 64.7849 |

### Sweep best cells
The default sweep uses the machine's preferred torch device and stores full results in `results/default_synthetic/cuda_sweep.csv`.

| Rank | Device | Gamma | Sigma | T | Mean terminal PnL | Sharpe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | mps | 1.00 | 0.50 | 2.0 | 115.8256 | 15.3746 |
| 2 | mps | 0.50 | 0.50 | 2.0 | 124.2544 | 14.6747 |
| 3 | mps | 0.10 | 0.50 | 2.0 | 129.8025 | 13.7046 |

## Replay / Calibration Notes
The replay engine intentionally remains a simplified top-of-book approximation. It does not model queue position, hidden liquidity, latency, or partial fill priority. The calibration step estimates `sigma`, `A`, `kappa`, and `epsilon` from the same abstraction, so the estimates should be interpreted as model-consistent rough inputs, not true microstructure calibration.

No real LOBSTER sample is present in `data/lobster/` right now. The replay and calibration code paths are verified by fixture-based tests under `tests/fixtures/`.

## Outputs
The default synthetic run writes:

- `results/default_synthetic/summary.json`
- `results/default_synthetic/pathwise_metrics.csv`
- `results/default_synthetic/baseline_comparison.csv`
- `results/default_synthetic/cuda_sweep.csv`
- `results/default_synthetic/avs_sweep_heatmap.png`

When replay data is present, calibration and backtest outputs are written into the same run directory as `calibration.json`, `replay_summary.json`, and `replay_inventory.csv`.
