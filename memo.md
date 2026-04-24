# P2 Research Memo — Avellaneda-Stoikov Market Maker

## Interview-Ready Conclusion
I implemented Avellaneda-Stoikov optimal quoting with a Glosten-Milgrom adverse-selection layer, validated it against one free-sample day of LOBSTER top-10 book data, and ablated it against four baseline quoters inside one shared execution engine. On the checked-in AAPL 2012-06-21 replay, the disciplined quote rules all reproduce essentially the same validated baseline: about `$662` terminal PnL on `972` fills with about `7.9%` replay spread capture. The main contribution is not a claim of deployable profitability. The contribution is a defensible research artifact: the HJB derivation, Ho-Stoll foundation, queue-reactive extension, and Glosten-Milgrom coupling are written up in `docs/hjb_derivation.md`; the synthetic and replay paths use one accounting convention; and the memo states plainly where realism still breaks, especially the one-day LOBSTER sample, uncalibrated `mu`, and missing joint queue-plus-information control.

## Introduction / Problem Statement
Avellaneda-Stoikov solves the canonical market-making control problem: a dealer wants to earn the bid-ask spread, but every passive fill changes inventory, and inventory is risky over the remaining trading horizon. That is a natural quantitative-research interview topic because it sits at the intersection of stochastic control, microstructure, simulation, and model skepticism. A candidate can derive the core HJB, implement the quoting rule, compare it against weaker baselines, and then explain what breaks when the stylized assumptions meet market data.

That is the lens for P2. The project is not trying to beat the market in one backtest. It is trying to answer the more defensible question: can I implement the Avellaneda-Stoikov quoting rule correctly, calibrate its primitives from a real top-of-book dataset, and explain how inventory risk, queue position, and adverse selection should interact in a realistic market-maker stack? Week 1 added a Glosten-Milgrom information layer, Week 2 added a cross-symbol ablation pipeline and replay smoke, and Week 3 packages the mathematics and the research readout into one coherent memo.

This memo therefore does three things. First, it states the control problem and points to the full derivation without duplicating it. Second, it summarizes what the synthetic engine and the LOBSTER replay actually show. Third, it records the limitations honestly: one checked-in day of AAPL is enough to validate implementation plumbing, not enough to claim robust alpha.

## Model
The full mathematics now live in `docs/hjb_derivation.md`. In words, the repo combines three layers. The first layer is the closed-form Avellaneda-Stoikov solution: Brownian mid-price, exponentially decaying fill intensities, reservation price `S - q * gamma * sigma^2 * (T - t)`, and optimal spread `gamma * sigma^2 * (T - t) + (2 / gamma) * log(1 + gamma / kappa)`. The second layer is a queue-reactive correction in the Huang-Lehalle-Rosenbaum style, implemented in `src/p2/queue.py`, where execution quality depends on queue depth and FIFO position rather than on quote distance alone. The third layer is a Glosten-Milgrom adverse-selection filter in `src/p2/glosten_milgrom.py`, where a fraction `mu` of traders are informed, order flow updates a posterior over latent value, and that posterior tilts effective buy and sell intensities. The memo uses the closed form as the base policy, the queue layer as a realistic correction, and the GM layer as the Week 1 information-sensitive extension.

Operationally, the most important implementation choice is that AVS and all baselines share one execution engine. That matters more than it sounds. It means differences in PnL or fill counts come from the quote logic rather than from inconsistent accounting, different adverse-selection handling, or different inventory clamps.

## Synthetic Simulator Check
The default synthetic run remains the cleanest implementation sanity check because every parameter is controlled:

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

This is still the right first check on the implementation. AVS earns slightly higher mean PnL than the simpler baselines, but the stronger point is that it cuts average absolute inventory roughly in half. That is the mechanism the theory is supposed to deliver. The reservation-price term does not exist to maximize raw spread capture mechanically; it exists to move the quote center away from the mid when inventory becomes risky.

The synthetic parameter sweep tells the same story in a broader grid. The best cells in `results/default_synthetic/cuda_sweep.csv` occur at lower volatility and longer horizon, where spread capture compounds while diffusion risk stays manageable. That sweep is useful for structural sensitivity analysis. It is not evidence that any specific parameter combination would survive real market frictions.

## Empirical Calibration on AAPL 2012-06-21
The checked-in replay artifact is built from the free LOBSTER AAPL top-10 sample stored under `data/lobster/`. All five quoter rows in `results/cross_symbol_ablation/cross_symbol_ablation.csv` use the same calibrated constants because they replay the same AAPL book and message files. The calibration estimates are:

| Date | Symbol | `sigma` | `A` | `kappa` | `epsilon` |
| --- | --- | ---: | ---: | ---: | ---: |
| 2012-06-21 | AAPL | 0.044287 | 0.258100 | 25.373720 | 0.001322 |

These numbers are informative, but they should be read narrowly.

`sigma = 0.044287` is an intraday top-of-book diffusion estimate, not a robust daily volatility parameter. `A = 0.258100` is a low baseline arrival scale because the replay abstraction only sees top-of-book fills and does not reconstruct full venue liquidity. `kappa = 25.373720` is steep, which means fill probability decays quickly with quote distance; small changes in placement matter a lot. `epsilon = 0.001322` is small in price units, but it still matters because replay PnL is also measured in small price increments per fill. The honest read is that these are usable local constants for one AAPL sample day, not portable market constants.

That last point matters for interpretation. Because all quoters share the same calibration and the replay is top-of-book only, the replay is better at checking internal consistency than at separating subtly different quoting policies. It is very good for answering "did the code use the same market assumptions across strategies?" It is not yet good for answering "which strategy is robustly better across days and symbols?"

That distinction is why the next data purchase would matter more than another round of parameter tuning. Multi-day LOBSTER would let the same calibration pipeline answer questions that the current artifact cannot: whether `kappa` is stable across sessions, whether `epsilon` changes materially around open and close, whether inventory-skewed quoters fail only on this AAPL path or systematically across names, and whether the random-quoter outperformance disappears once luck is averaged out. Without that panel dimension, the right scientific stance is implementation validation, not performance extrapolation.
That is the difference between a credible research memo and a one-day anecdote.

## Cross-Symbol Ablation (Week 2 Deliverable)
The Week 2 design originally aimed for a multi-day robustness panel. The free LOBSTER constraint forced a methodology pivot documented in `docs/ablation_report.md`: use a shared date across symbols when data are available, and treat the result as a cross-sectional sanity check rather than a time-series robustness claim. In the checked-in repo artifact, the pipeline is present, but the recorded result is the AAPL smoke only: five rows, one per quoter, all for `2012-06-21`.

The comparison table below pulls the actual metrics from `results/cross_symbol_ablation/cross_symbol_ablation.csv`.

| Quoter | Sharpe | Spread capture | Avg abs inventory | Bid fill rate | Ask fill rate | Replay terminal PnL | Replay spread capture | Replay fills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `avs_optimal` | 0.2871 | 0.006931 | 0.080368 | 0.088 | 0.088 | 662.07 | 7.87% | 972 |
| `constant_spread` | 0.3376 | 0.007926 | 0.100761 | 0.107 | 0.094 | 662.17 | 7.89% | 972 |
| `inventory_linear` | 0.3379 | 0.006901 | 0.082401 | 0.093 | 0.082 | 13.55 | 7.89% | 846 |
| `random` | 0.2676 | 0.007781 | 0.095107 | 0.124 | 0.100 | 742.95 | 6.11% | 1207 |
| `symmetric` | 0.3231 | 0.006348 | 0.084112 | 0.082 | 0.079 | 662.17 | 7.89% | 972 |

Three points matter more than the raw ranking.

First, AVS-optimal, symmetric, and constant-spread all collapse onto the same replay baseline: about `$662` terminal PnL on `972` fills with `7.9%` replay spread capture. That is not a failure of the experiment. It is a clue about the model and the data. All three quote rules are disciplined, all three use the same AAPL calibration, and the replay itself is top-of-book only with no queue priority or latency. Under those conditions, it is unsurprising that they land on essentially the same realized path.

Second, `InventoryLinearMM` and `RandomQuoter` diverge for reasons that are easy to explain from the code. `InventoryLinearMM` uses a linear skew with `lambda_q = 1` in `src/p2/baselines.py`. Once `|q| > 0`, it widens one side and narrows the other aggressively. In replay that means it often becomes too one-sided too early, which protects inventory but gives up fills. The artifact shows exactly that failure mode: replay fills fall to `846`, and replay terminal PnL collapses to `$13.55`. The low replay average absolute inventory, `2.27`, is not evidence of superiority; it is evidence that the strategy stops trading the moment inventory pressure appears.

Third, `RandomQuoter` posts the highest replay PnL, `$742.95`, but that is not an investable conclusion. The same row has the worst replay spread capture, `6.11%`, the highest replay average absolute inventory, `63.96`, and the most fills, `1207`. The correct reading is path luck. On this one day, random quote placement happened to harvest more gross PnL, but it did so with weaker spread quality and much larger inventory exposure. That is the opposite of the kind of result I would defend in an interview as a stable edge.

The synthetic metrics in the same table should also be interpreted carefully. The Sharpe values, `0.268` to `0.338`, are close enough that I do not treat the ordering as meaningful on one AAPL calibration. They show that the five quoters live in the same rough operating regime under the shared simulator. They do not establish a robust cross-symbol winner. The honest conclusion for Week 2 is therefore narrow: the ablation pipeline works, the AAPL smoke reproduces the validated baseline, and the real ranking exercise still requires paid multi-day LOBSTER or a richer dataset such as ITCH or TAQ.

That is still a useful result in interview terms. Many market-making projects fail at the boring part: the baseline cannot be reproduced twice, different strategies use slightly different execution rules, or the writeup quietly avoids explaining why a naive control beat the supposedly optimal one. This artifact does the opposite. It shows exactly where the disciplined quoters agree, exactly how the crude quoters fail, and exactly why one lucky random path is not enough to override microstructure intuition. That is a stronger research posture than presenting a noisier but less interpretable leaderboard.

## Glosten-Milgrom Adverse-Selection Sensitivity (Week 1 Deliverable)
The Week 1 extension adds the information asymmetry that the plain Avellaneda-Stoikov model leaves out. In `src/p2/glosten_milgrom.py`, a fraction `mu` of traders is informed about a latent binary value state. The market maker observes the sign of incoming trades and updates the posterior probability of the high-value state via Bayes.

The clean way to understand the update is in log-odds form. Let
`p_t = P(V = v_high | order flow up to t)` and
`ell_t = log(p_t / (1 - p_t))`. A buy order adds
`log((1 + mu) / (1 - mu))` to `ell_t`; a sell order subtracts the same amount. As
`mu` rises, each trade carries more information, so the posterior swings faster
for the same observed order-flow imbalance. That is exactly the adverse-selection
story market makers care about: when flow is more informed, a streak of buys is
not just inventory flow, it is evidence that your stale ask is too cheap.

The repo currently couples that logic to AVS through intensity adjustment:

- `lambda_buy = lambda_baseline * (1 - mu + 2 * mu * p_t)`
- `lambda_sell = lambda_baseline * (1 - mu + 2 * mu * (1 - p_t))`

When `p_t = 0.5`, the adjustment disappears and the model falls back to the plain
AvS intensity. When `p_t > 0.5`, buy-side flow is more likely, so ask fills become
more toxic and the dealer's subjective fair value shifts upward. In the language
of the derivation appendix, the AVS reservation price and the GM posterior mean
act on the same object. Inventory risk shifts the reservation price by
`-q * gamma * sigma^2 * (T - t)`. Adverse selection shifts it by the posterior
mean relative to the mid. Those effects are additive in the quoting rule even
though they come from different economic channels.

The practical sensitivity claim is therefore straightforward. As `mu` increases,
the posterior responds more sharply to the same trade sequence, the effective
intensities become more one-sided, and the economically sensible quoting response
is more conservative. In a fully solved coupled HJB, that means a more skewed
reservation price and effectively wider exposure against informed flow. In this
repo, `mu` is still a scenario variable rather than a calibrated parameter, so the
GM layer is best understood as a structurally correct adverse-selection extension,
not yet as a production-ready estimate of toxicity.

## Week 2 — 5-scenario regime sweep
To keep the Week 2 "week-like" distribution story honest despite having only one checked-in LOBSTER day, I added a five-scenario sweep under `results/lobster_week_sweep/`. `scenario_0_base` is replay-anchored to the AAPL `2012-06-21` calibration and is forced to stay within a documented `+/-15%` tolerance of the checked-in baseline. Scenarios `1` through `4` are synthetic perturbations of that same calibration on the same AAPL event-time grid. They are not independent LOBSTER days.

The sweep uses deterministic seeds `20260424` through `20260428` and reports cumulative PnL curves, inventory paths, max drawdown, per-minute Sharpe, and per-minute information ratio. The selected replay-consistent `gamma` is `0.006`, which reproduces the existing anchor almost exactly: `scenario_0_base` finishes at `$662.16` on `972` fills with `7.88%` spread capture, versus the checked-in `$662.07 / 972 / 7.87%`.

| Scenario | Construction | Terminal PnL | Max DD | Sharpe | IR |
| --- | --- | ---: | ---: | ---: | ---: |
| `scenario_0_base` | AAPL replay anchor | 662.16 | 123.83 | 0.1242 | 0.1241 |
| `scenario_1_high_vol` | `2x sigma` | 411.67 | 100.30 | 0.1281 | 0.1281 |
| `scenario_2_low_vol` | `0.5x sigma` | 662.15 | 123.83 | 0.1242 | 0.1241 |
| `scenario_3_thin_book` | `0.5x A`, `2x kappa` | 65.43 | 25.63 | 0.1035 | 0.1038 |
| `scenario_4_adverse_selection` | `pi = 0.30` vs `0.10` | 627.90 | 177.16 | 0.1175 | 0.1176 |

Three reads matter. First, the base anchor now gives the week story a defensible empirical reference point instead of pretending that five real LOBSTER days exist on disk. Second, the thin-book stress is the harshest liquidity shock: fills fall from `972` to `579`, spread capture is cut roughly in half, and terminal PnL drops to about `$65`. Third, the adverse-selection stress does not destroy gross PnL, but it produces the worst drawdown in the panel, `177.16`, which is exactly the kind of toxicity-sensitive behavior the Glosten-Milgrom extension is supposed to surface.

The low-vol scenario lands almost on top of the base case. I would not over-interpret that as a robust invariance claim. It is a consequence of running a replay-anchored event-time sweep on one AAPL day where the quote threshold is already close to the historical trade prices. In other words, the sweep is useful as a regime-stress narrative and a sensitivity panel, not as evidence that calm-market performance is fully pinned down by one free-sample replay.

## Limitations
The main limitations are structural and should be stated directly.

- The checked-in LOBSTER evidence is one trading day, `2012-06-21`, for one symbol, AAPL. That is enough to validate the replay and calibration path, not enough to claim cross-day robustness.
- The replay is top-of-book only. There is no venue fragmentation, latency model, partial-fill model, or true queue-priority reconstruction.
- The queue-reactive machinery exists in `src/p2/queue.py`, but it is not yet jointly solved with the Glosten-Milgrom information layer inside one coupled control problem.
- The GM informed-trader fraction `mu` is not calibrated from data in this session. It is treated as a scenario parameter.
- Independent Poisson fill intensities remain a stylized approximation even after the queue and GM corrections.
- Spread capture in simulation is still optimistic relative to a live venue with competing market makers, cancellations, and stale-quote risk.
- None of these artifacts should be read as evidence of deployable profitability.

That is the right boundary for the project. The repo is strong as a QR interview artifact because it combines mathematical derivation, reproducible code, and an honest account of what the replay can and cannot support. It would stop being strong if it pretended that one free-sample AAPL day settled the strategy question.
