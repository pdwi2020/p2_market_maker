# Ambiguity Log

## Half-Spread Convention
The paper writes the optimal total spread. The repo uses the one-sided half-spread for quote placement. This is intentional and is documented in `hjb_solver.py`, `README.md`, and `src/p2/derivation.md`.

## Inventory Limits
The closed-form AVS solution does not include hard inventory cutoffs. The repo adds them as an execution-layer control by disabling the relevant quote side at the bound. This is a practical extension, not part of the analytical derivation.

## Adverse Selection
The analytical model does not include post-fill price jumps. The repo adds a configurable `epsilon` jump after fills so the synthetic engine can represent a basic adverse-selection penalty.

## Replay Fidelity
The LOBSTER replay path is top-of-book only. It does not model queue position, venue latency, or hidden liquidity, so replay results should be read as qualitative stress tests rather than production backtests.

## Calibration Claims
Because replay is simplified, the estimated `sigma`, `A`, `kappa`, and `epsilon` are model-consistent coarse inputs, not true microstructure calibration outputs.
