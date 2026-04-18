# P2 Research Memo — Avellaneda-Stoikov Market Maker

## Objective
The goal of P2 is to turn the Avellaneda-Stoikov market-making model into a defensible research project rather than a formula demo. The repo now covers the analytical HJB solution, a reproducible synthetic simulator, baseline comparisons, a parameter sweep, and a replay/calibration path for free LOBSTER samples when those files are available locally.

The deliverable is aimed at prop and market-making interview contexts: it should show that the model is implemented correctly, that inventory and fill mechanics are handled consistently, and that the limitations of the replay abstraction are understood rather than hidden.

## Model and Implementation
The analytical core is unchanged from Avellaneda-Stoikov:

- Mid price follows arithmetic Brownian motion.
- Order arrivals decay exponentially with quote distance.
- Reservation price is `S - q * gamma * sigma^2 * (T - t)`.
- Total spread is `gamma * sigma^2 * (T - t) + (2 / gamma) * log(1 + gamma / kappa)`.

The implementation details that matter operationally are:

1. The repo uses one shared execution engine for AVS and the baseline strategies, so the comparison is not contaminated by different fill logic.
2. Inventory hard limits are enforced by suppressing the relevant quote side once the limit is reached. This avoids pretending that post-fill clipping is a realistic control rule.
3. Adverse selection is modeled as an immediate mid-price jump of `+epsilon` after ask fills and `-epsilon` after bid fills.
4. All outputs are driven from `configs/p2_config.yaml` and written into `results/<run_name>/`.

The replay engine is intentionally narrower than the synthetic engine. It is top-of-book only, does not model queue priority or latency, and should be treated as a qualitative stress test rather than a production backtest.

## Synthetic Results
The default synthetic run was generated with:

- `sigma = 1.0`
- `gamma = 0.1`
- `kappa = 1.5`
- `A = 140.0`
- `T = 1.0`
- `dt = 0.001`
- `Q_max = 10`
- `epsilon = 0.02`
- `n_paths = 1000`
- seed `42`

Results from `results/default_synthetic/summary.json`:

| Strategy | Mean terminal PnL | PnL std | Sharpe | Avg abs inventory | Spread capture |
| --- | ---: | ---: | ---: | ---: | ---: |
| AVS | 64.5116 | 6.6831 | 9.6530 | 2.0828 | 67.7794 |
| Symmetric | 60.1049 | 7.7000 | 7.8058 | 4.4297 | 61.7990 |
| Constant spread | 63.4168 | 8.5089 | 7.4530 | 4.2453 | 64.7849 |

The key result is not simply that AVS earns the most mean PnL. It also controls inventory materially better than the symmetric and constant-spread baselines while preserving higher realized spread capture per unit of risk. The average absolute inventory is roughly half that of the two simpler baselines, which is exactly the mechanism the reservation-price shift is supposed to provide.

This is synthetic evidence, not live edge. The importance is methodological: the research repo now demonstrates correct implementation, consistent accounting, and controlled baseline comparison.

## Parameter Sweep
The sweep uses the same inventory rules, quote equations, and adverse-selection logic as the default simulator. On this machine it ran on `mps` and produced `results/default_synthetic/cuda_sweep.csv` together with `results/default_synthetic/avs_sweep_heatmap.png`.

The best sweep cells were:

| Rank | Device | Gamma | Sigma | T | Mean terminal PnL | Sharpe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | mps | 1.00 | 0.50 | 2.0 | 115.8256 | 15.3746 |
| 2 | mps | 0.50 | 0.50 | 2.0 | 124.2544 | 14.6747 |
| 3 | mps | 0.10 | 0.50 | 2.0 | 129.8025 | 13.7046 |

The broad pattern is intuitive:

- Lower volatility and longer horizon improve synthetic Sharpe in this setup because spread capture compounds while diffusion risk remains manageable.
- Higher `gamma` reduces inventory excursions and can improve risk-adjusted performance even when it reduces aggressiveness.
- The model remains sensitive to the chosen fill-intensity scale `A`, so the sweep should be read as structural sensitivity analysis, not a parameter search for a live strategy.

## Replay and Calibration Status
No real LOBSTER sample is currently stored under `data/lobster/`. The repo therefore behaves as follows:

- `make calibrate` fails with a direct missing-data message rather than silently fabricating inputs.
- `make backtest` fails with the same direct message.
- Replay and calibration logic are still exercised by fixture-based tests under `tests/fixtures/`.

The calibration path estimates:

- `sigma` from top-of-book mid-price changes
- `A` and `kappa` from empirical trade intensity versus distance from mid
- `epsilon` from next-step adverse movement after trade-direction events

Those estimates are intentionally simple because the replay model itself is simple. It would be misleading to claim full microstructure calibration without queue state, venue fragmentation, latency, and partial fill modeling.

## What Did Not Work
Several issues surfaced during implementation and were fixed explicitly:

1. The original repo used a repo-local `.venv`, which conflicted with the drive-level environment contract.
2. The original sweep path used a simplified torch simulation that did not match the core synthetic engine semantics.
3. The shared drive environment had broken `matplotlib` metadata and ExFAT `._*` sidecar files that prevented plotting imports until the packages were reinstalled and cleaned.
4. The derivation and memo were placeholder outlines, which made the repo look unfinished despite a decent code scaffold.

There is still one unresolved infrastructure tension: the drive-level root notes mention Python `3.11 via uv`, while the operational shared environment used here is Python `3.14.3`. The repo runs correctly now, but that version-policy mismatch should be cleaned up later.

## Limitations
The main limitations are structural, not hidden:

- Independent Poisson bid and ask arrivals are a stylized approximation.
- There is no queue-position model.
- Replay is top-of-book only.
- Spread capture in the simulator is optimistic relative to a real venue with latency, competition, and cancellations.
- The default synthetic run is useful for replication and reasoning, not for claiming deployable profitability.

## Conclusion
P2 is now in the right shape for interview discussion. It shows:

- the HJB solution is implemented correctly
- the simulator, baselines, and sweep share one consistent set of assumptions
- inventory control is handled explicitly
- replay and calibration exist, but their limitations are stated clearly

That is the right level of rigor for a market-making project in a QR portfolio: strong analytical grounding, reproducible code, real artifacts, and no false claims about the realism of the backtest.
