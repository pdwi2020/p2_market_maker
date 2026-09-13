# Market-Making Study Preregistration

## Research question

Does queue-aware, inventory-aware quoting improve out-of-sample risk-adjusted
net PnL relative to symmetric touch quoting after fixed fees, latency, inventory
limits, and post-only execution are applied?

The confirmatory result is the BTCUSDT perpetual study in the locked test
window. The ETHUSDT, SOLUSDT, ES, external-engine, and LOBSTER results are
robustness or validation studies.

## Data split and walk-forward protocol

- Selection window: 2025-05-01 through 2025-06-30 UTC.
- Locked test window: 2025-07-01 through 2026-04-27 UTC.
- Parameters for trading date `d` are calibrated only from date `d-1`.
- A date is skipped if its preceding calendar date is unavailable, has an
  invalid book interval, or cannot produce a valid calibration.
- Strategy parameters are selected once on the selection window and are not
  changed after the test-window replay begins.
- BTCUSDT uses every eligible test date. ETHUSDT and SOLUSDT use the first
  available Wednesday in each ISO week; if Wednesday is unavailable, the next
  available date in that ISO week is used.

The selection criterion within each parameterized strategy family is the
highest annualized Sharpe ratio of daily net PnL in the selection window.
Ties, after rounding Sharpe to six decimal places, are resolved by higher mean
daily net PnL and then by the lower numeric parameter tuple. The four selected
strategies are evaluated without further tuning in the locked test window.

## Contract and replay rules

The primary contract is the Bybit BTCUSDT perpetual. All timestamps and daily
boundaries are UTC.

| Setting | Fixed value |
| --- | ---: |
| Quote size | 0.01 BTC |
| Tick size | 0.1 USDT |
| Minimum time between quote decisions | 100 ms |
| Inventory bound | -0.05 to 0.05 BTC |
| End-of-day action | No flattening trade |
| Primary maker fee | 0.020% of fill notional |
| Taker fee | 0.055% of fill notional |
| Primary latency | 50 ms |

The replay processes book updates before trades when their exchange timestamps
are equal. A quote decision may be made after any event, but a changed quote is
submitted only when at least 100 ms has elapsed since the previous submission.
Submission, cancellation, and replacement become effective after the latency
delay. A replacement loses queue priority. A newly effective order joins the
back of displayed volume at its price.

Orders are post-only. A bid that would be marketable when it becomes effective
is repriced to one tick below the current ask; an ask is handled symmetrically.
Orders that would increase absolute inventory are suppressed at the inventory
bound. Partial fills are permitted, and no position is flattened at day end.
Working orders are canceled at the end of the UTC date. Funding payments are
not included because the input event stream does not contain funding records.

Daily net PnL is end cash plus end inventory marked at the final valid midpoint,
less initial cash. Fees are applied to absolute fill notional. A negative maker
rate is recorded as a credit. The taker rate is implemented for completeness;
the preregistered post-only paths are expected to produce maker fills only.

## Calibration

Calibration for date `d` uses only the reconstructed events from date `d-1`.

### Volatility

The midpoint is sampled at the last valid observation in each UTC second.
Seconds following an invalid-book gap are omitted until the next snapshot.
Sigma is the sample standard deviation of consecutive one-second midpoint
changes, in USDT per square-root second. At least 3,600 valid changes are
required.

### Trade-arrival intensity

For each trade, depth is measured from the immediately preceding midpoint in
the aggressor direction. The fixed depth grid is

`delta = 0.05, 0.15, ..., 1.95 USDT` (20 points).

At each depth, lambda is the number of trades reaching or passing that depth,
divided by valid observed seconds. Ordinary least squares fits

`log(lambda(delta)) = log(A) - kappa * delta`.

Only positive-intensity grid points are used. A calibration is valid when at
least five points remain, the fitted `A` is positive, `kappa` is positive, and
sigma is finite and positive. No test-date information is used as a fallback.

## Strategy definitions and selection trials

Inventory `q` is in BTC, midpoint `s` and quotes are in USDT, quote size
`Delta` is 0.01 BTC, and calibrated intensity is
`lambda(delta) = A exp(-kappa delta)`. Bid prices are rounded down and ask
prices up to the tick before post-only handling.

### 1. Symmetric touch

Join the current best bid and best ask. This strategy has no fitted quoting
parameter and contributes one selection trial.

### 2. Avellaneda-Stoikov

The rolling session horizon is `T = 86,400 seconds`, with
`tau = max(T - elapsed_session_seconds, 0)`. Reservation price and half-spread
are

`r = s - q * gamma * sigma^2 * tau`

`h = gamma * sigma^2 * tau / 2 + log(1 + gamma / kappa) / gamma`.

The raw quotes are `r - h` and `r + h`. The risk-aversion grid is

`gamma = {0.00001, 0.00005, 0.0001, 0.0005, 0.001}`.

This contributes five selection trials.

### 3. GLFT asymptotic quotes

The finite-order-size closed-form approximation uses

`c1 = log(1 + gamma * Delta / kappa) / (gamma * Delta)`

`c2 = sqrt(gamma / (2 * A * Delta * kappa)
           * (1 + gamma * Delta / kappa)^(kappa / (gamma * Delta) + 1))`

`r = s - q * sigma * c2`

`h = c1 + Delta * sigma * c2 / 2`.

The raw quotes are `r - h` and `r + h`, subject to the inventory bounds. The
same five-value `gamma` grid is used, contributing five selection trials.

### 4. GLFT with touch imbalance

Touch imbalance is

`I = (bid_size - ask_size) / (bid_size + ask_size)`.

Both GLFT quotes are shifted by `beta * tick_size * I`. Positive imbalance
therefore raises both quotes. The `gamma` grid is unchanged and

`beta = {0.5, 1.0, 2.0, 4.0}` ticks.

This contributes 5 x 4 = 20 selection trials.

The total preregistered selection count is therefore

`N = 1 + 5 + 5 + 20 = 31`.

The Deflated Sharpe Ratio uses `N = 31`, the variance of the 31 selection-window
trial Sharpes, and the skewness and kurtosis of the selected daily-PnL series.
Execution-cost sensitivities below are not used for selection and do not
increase this value of `N`.

## Cost and latency sensitivity grid

The maker-rate grid is

`{0.020%, 0.010%, 0.000%, -0.005%}`.

The latency grid is `{10, 50, 200} ms`. The taker fee remains 0.055%. These
settings form 4 x 3 = 12 sensitivity cells per locked strategy and 48 reported
strategy-cell combinations. The primary table uses 0.020% maker fees and
50 ms latency. The complete cross-product over all 31 selection candidates
would contain 372 cells, but it is not searched and is not used in the DSR
trial count.

## Study A: out-of-sample crypto replay

For each strategy and symbol, report:

- per-day gross and net PnL in USDT;
- annualized Sharpe of daily net PnL using `sqrt(365)`;
- a 95% moving-block bootstrap confidence interval for Sharpe, with seven-day
  blocks, 10,000 resamples, and seed 20250701;
- absolute peak-to-trough maximum drawdown of cumulative daily net PnL;
- filled volume divided by activated quote volume;
- activated quote count divided by fill count;
- one-second sampled mean absolute inventory, inventory standard deviation,
  maximum absolute inventory, and end-of-day inventory;
- realized spread, inventory revaluation, fees, and net PnL; and
- signed midpoint markouts at 1, 5, 30, and 60 seconds.

For a fill with signed inventory change `dq`, realized spread is
`-dq * (fill_price - midpoint_at_fill)`. Inventory revaluation is
`dq * (final_midpoint - midpoint_at_fill)`. Net PnL must equal realized spread
plus inventory revaluation minus fees. The signed markout at horizon `h` is
`dq * (midpoint_at_fill+h - midpoint_at_fill)`; a horizon beyond the final
valid midpoint is omitted.

## Study B: ES queue-model validation

Use ES.c.0 during 14:30-21:00 UTC on 2025-01-06 and 2025-01-07. The symmetric
strategy joins the touch with one contract, a five-contract inventory bound,
100 ms minimum requote interval, 10 ms latency, and zero fees so the comparison
isolates fill modeling. Compare three methods:

1. exact order-level FIFO fills;
2. level-two queue approximation with cancel-from-back; and
3. level-two queue approximation with proportional cancellation.

Report fill count, fill volume, fill-time distribution, 1/5/30/60-second
markouts, and gross PnL by method. The approximation with the smallest absolute
relative fill-count bias versus exact FIFO is selected for Study A. A tie is
resolved first by the lower two-sample Kolmogorov-Smirnov distance between
fill-time distributions, then by lower absolute gross-PnL error, and finally in
favor of proportional cancellation. This is a conditional rule fixed before
the crypto results are generated.

## Study C: external replay cross-check

Run symmetric touch and the selected GLFT strategy through the independently
installed `hftbacktest` package on these five BTCUSDT dates:

- 2025-07-07
- 2025-08-04
- 2025-10-06
- 2026-01-05
- 2026-04-06

Use the primary fee and latency settings. Report fill count, filled volume, net
PnL, and absolute and relative differences from the native replay. The package
is an external dependency; no source or examples from it are copied.

## Study D: LOBSTER appendix

Run AAPL, AMZN, GOOG, INTC, and MSFT for the public 2012-06-21 sample, skipping
unavailable symbols. Each symbol uses its first 30 minutes for same-day
calibration and the remainder for the appendix replay; this appendix is not
part of the confirmatory split. Use all four strategy families with the
BTC-selected `gamma` and `beta`, one-share quotes, a $0.01 tick, a ten-share
inventory bound, 100 ms minimum requote interval, 1 ms latency, a $0.0025 per
share maker rebate, and a $0.0030 per share taker fee.

## Output contract

One `make research` invocation with the full-history directory configured must
regenerate only these published artifacts:

- `summary.json`
- `daily_pnl.csv`
- `decomposition.csv`
- `markouts.csv`
- `queue_validation.csv`
- `hftbacktest_crosscheck.csv`
- `lobster_appendix.csv`
- cumulative net-PnL figure by strategy
- markout-curve figure
- PnL-decomposition figure
- inventory-distribution figure
- queue-model-error figure

Figures read the published tables rather than private replay state. Tabular
outputs contain numbers and identifiers only. Missing optional LOBSTER symbols
are listed in `summary.json`.

## References

- Avellaneda, M. and Stoikov, S. (2008), *High-frequency trading in a limit
  order book*, Quantitative Finance 8(3), 217-224.
- Guéant, O., Lehalle, C.-A., and Fernandez-Tapia, J. (2013), *Dealing with the
  inventory risk: a solution to the market making problem*, Mathematics and
  Financial Economics 7, 477-507. <https://doi.org/10.1007/s11579-012-0087-0>
- Cont, R., Kukanov, A., and Stoikov, S. (2014), *The price impact of order book
  events*, Journal of Financial Econometrics 12(1), 47-88.
  <https://arxiv.org/abs/1011.6402>
- Bailey, D. H. and López de Prado, M. (2014), *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality*.
  <https://doi.org/10.2139/ssrn.2460551>
- `hftbacktest` project documentation: <https://hftbacktest.readthedocs.io/>
