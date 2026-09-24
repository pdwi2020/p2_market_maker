# Market Making on Real Limit Order Books: A Preregistered Negative Result

## Abstract

Four quoting rules were preregistered and tested on 301 days of Bybit BTCUSDT
perpetual order-book data, with robustness runs on ETHUSDT and SOLUSDT, queue
validation against order-level CME futures data, an independent replay engine,
and an equity appendix. All four lose money, at every fee and latency setting in
the grid, with annualised Sharpe ratios between -6.5 and -33.8 and a deflated
Sharpe ratio of zero across 31 trials. The loss is not primarily the exchange
fee. On BTCUSDT the quoted spread is about 0.01 basis points of notional, while
one-second adverse selection costs 0.30 to 0.41 basis points and the VIP0 maker
fee is 2.0 basis points. The spread is roughly two orders of magnitude too thin
to pay for either. The same engine on a 2012 AAPL sample, where the quoted width
is roughly two hundred times wider and the maker side earns a rebate, turns
positive on the one day tested. That single control cannot carry a causal claim
on its own, but together with the decomposition it points at the venue's
spread-to-cost ratio rather than at the quoting rules.

## Question

Fixed before any result was computed, in `docs/preregistration.md`:

> On real limit order books, does inventory-aware quoting (Avellaneda-Stoikov,
> GLFT) plus a short-horizon order-flow skew earn positive net PnL after queue
> position, latency, fees and adverse selection? How much of the naive spread
> capture do markouts eat?

The preregistration fixes the strategies and their parameter grids, the
calibration protocol, the walk-forward split, the cost and latency grid, the
metrics, and the rule for choosing between queue approximations. Three
amendments were made after the study was built, each recorded with the
measurement that prompted it and a before-and-after table. None of them was
made after seeing a test-window result.

## Data

| Source | Coverage | Role |
| --- | --- | --- |
| Bybit perpetual L2 book and trades | BTCUSDT, ETHUSDT, SOLUSDT, 2025-05-01 to 2026-04-27 | Main study |
| Databento CME MBO, ES.c.0 | 2025-01-06 and 2025-01-07, order level | Queue validation |
| LOBSTER level-10 sample, AAPL | 2012-06-21 | Equity appendix |

The Bybit feed arrives as snapshot and delta messages carrying 200 price levels
per side. Reconstruction replays them into a price-keyed book, requires
`update_id` to advance by one between deltas, marks the book invalid from any
gap until the next snapshot, asserts the book is never crossed, and caches the
top 20 levels per side merged with the trade stream. Across all 532
reconstructed symbol-days the crossed-book count is zero, there are no
sequence gaps, and 870 seconds in total are marked invalid.

Parameters for trading day `d` come only from day `d-1`: sigma from one-second
midpoint changes, and `A` and `kappa` from an ordinary least squares fit of
`log lambda(delta) = log A - kappa delta` over a fixed 20-point depth grid. A
date without a valid preceding-day calibration is excluded and recorded; 27 test
dates were excluded this way, all of them SOLUSDT, which is why SOLUSDT
contributes 16 days against ETHUSDT's 43.

## Engine and fill model

The replay is event driven. Quotes are post-only: one that would be marketable
when it becomes effective is repriced one tick inside the opposite touch, so no
preregistered path can take liquidity, and the published marketable-fill count
is zero everywhere. Submission, cancellation and replacement take effect after
50 ms, and a replacement loses queue priority. Quotes are reconsidered at most
every 100 ms. Inventory is bounded to plus or minus 0.05 BTC by suppressing the
side that would breach it, and nothing is flattened at day end.

Two parts of the fill model deserve stating explicitly, because both were
corrected during the build and both materially change the answer.

**Fills follow traded volume.** Bybit prints one row per matched price level,
best price first: measured over three sample days, 15 to 19 per cent of
same-millisecond same-side trade groups span more than one price, with a median
of five rows. Aggressor volume is therefore observable level by level. A trade
at or through our price clears the queue ahead using its own volume and fills
only the residual. A genuine sweep still fills the order completely, but through
volume that actually traded rather than by assumption.

**Placement has three states.** A quote price carrying displayed size puts us
behind it. A price inside the reconstructed window with no displayed size leaves
us alone at the front of an empty level. A price deeper than the deepest
reconstructed level has an unknown queue, and there the replay refuses to claim
a fill at all rather than inventing a position. The three counts are published
per strategy-day, and they matter: 88 per cent of GLFT quotes rest on empty
levels inside the book, while 99 per cent of Avellaneda-Stoikov quotes fall
beyond it.

## Results

Selection ran on 2025-05-01 to 2025-06-30 across 31 trials and locked four
strategies. Every one of the 31 trials had a negative selection-window Sharpe,
ranging from -9.7 to -45.1, so the selection step chose the least bad rather
than a promising one. The test window, 2025-07-01 to 2026-04-27, was then run
once.

BTCUSDT, 301 days, maker 0.020 per cent, latency 50 ms:

| Strategy | Net PnL | Ann. Sharpe | 95% CI | DSR | Fills | Gross, bp | Profitable days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| symmetric touch | -1,548,538 | -33.75 | [-39.87, -30.06] | 0.00 | 16,489,122 | -1.29 | 0 |
| GLFT, gamma 1e-4 | -1,077,783 | -30.20 | [-36.52, -26.71] | 0.00 | 13,826,214 | -1.23 | 0 |
| GLFT + imbalance, beta 2 | -1,080,896 | -29.96 | [-36.26, -26.49] | 0.00 | 13,765,443 | -1.24 | 0 |
| Avellaneda-Stoikov, gamma 1e-3 | -5,915 | -6.53 | [-8.70, -5.51] | 0.00 | 68,870 | -1.52 | 17 |

Three of the four lose money on every single one of the 301 days. This is not a
strategy with a bad Sharpe; it is a strategy with a deterministic leak. ETHUSDT
and SOLUSDT reproduce the pattern on their weekly samples, with symmetric at
-40.2 and -44.9 annualised Sharpe respectively.

The decomposition shows where it goes:

| Strategy | Realized spread | Inventory revaluation | Fees | Net |
| --- | ---: | ---: | ---: | ---: |
| symmetric touch | -78,217 | -567,866 | 902,455 | -1,548,538 |
| GLFT, gamma 1e-4 | 58,955 | -492,981 | 643,758 | -1,077,783 |
| GLFT + imbalance | 60,528 | -497,175 | 644,249 | -1,080,896 |
| Avellaneda-Stoikov | 659 | -3,335 | 3,239 | -5,915 |

Fees are published as a positive magnitude, so net is realized spread plus
inventory revaluation minus fees.

The symmetric quoter's realized spread is negative, at -0.047 basis points per
fill. A quoter joining the touch should buy below the midpoint and sell above
it, so a negative realized spread is worth pausing on. It is the signature of
being picked off: with a 100 ms requote floor and 50 ms of latency, the fills it
receives are disproportionately those where the midpoint has already moved
through its quote. GLFT rests about 4 USDT off the midpoint and realises only
+0.043 basis points, meaning the midpoint has travelled almost the whole way to
its quote by the time it trades. Signed markouts confirm this directly: negative
at every horizon for every strategy, with most of the damage inside one second.

Set against those numbers, the BTCUSDT quoted spread is about 0.01 basis points.
The arithmetic is not close.

## Queue-model validation

The crypto book is level-two data, so queue position is modelled. The ES study
replays the same quoting logic against order-level MBO, where a hypothetical
resting order's fill times follow exactly from the fills and cancels ahead of
it, and compares that truth against the two level-two approximations. Four arms
were run: the preregistered touch arm at 10 ms, the same at 50 ms to match the
crypto study, and quotes two and eight ticks off the midpoint at 50 ms, because
a validation that never leaves the touch says nothing about strategies that
quote away from it.

| Arm | Method | Fills | Fill-count bias | Gross PnL | Error vs exact |
| --- | --- | ---: | ---: | ---: | ---: |
| touch, 10 ms | exact FIFO | 68,478 | - | -73,262 | - |
| touch, 10 ms | L2 cancel-from-back | 68,774 | +0.4% | -82,312 | -9,050 |
| touch, 10 ms | L2 proportional | 93,363 | +36.3% | -76,894 | -3,631 |
| touch, 50 ms | exact FIFO | 63,198 | - | -75,462 | - |
| touch, 50 ms | L2 cancel-from-back | 63,372 | +0.3% | -84,487 | -9,025 |
| 2 ticks off, 50 ms | exact FIFO | 1,336 | - | +3,175 | - |
| 2 ticks off, 50 ms | L2 cancel-from-back | 1,400 | +4.8% | +1,187 | -1,987 |

Cancel-from-back was chosen by the preregistered rule on the preregistered arm,
before the crypto study ran. It overstates the gross loss by 12.4 per cent of
the exact value at the touch. The honest reading is that the headline losses are
biased away from zero by roughly an eighth, which does not come close to
changing the conclusion. The eight-ticks-off arm produced twelve fills across
two days, which is itself informative: on ES, quoting that far out essentially
never trades.

## Independent engine

The same five days were replayed through the external `hftbacktest` package with
queue models matched, both advancing a resting order only on traded volume.
Fill counts differ by 4.7 to 32.2 per cent on the symmetric strategy and 35.9 to
66.0 per cent on GLFT, with the external engine consistently more conservative.
One modelling difference remains and explains the direction: the external
exchange fills our single 0.01 BTC order all or nothing, while the native engine
permits a partial fill. That bites hardest on GLFT, which rests away from the
touch where passing volume is thinner. Both engines agree on sign and scale.

## Robustness

Twelve cells were run: maker fees of -0.005, 0.000, 0.010 and 0.020 per cent
against latencies of 10, 50 and 200 milliseconds. None is profitable. The best
cell in the entire grid is the symmetric quoter at the rebate tier with 200 ms
latency, at -387,341. At a zero fee the same strategy still loses 646,083 gross,
so the exchange fee aggravates the result without causing it.

Latency behaves counterintuitively: more latency loses less. That is consistent
with the pick-off story, since a slower quoter both trades less and keeps stale
quotes that the market has already left, rather than chasing a midpoint that is
moving away from it.

## Equity appendix

The same engine on the public AAPL sample for 2012-06-21, under 2012 Nasdaq
maker-rebate assumptions:

| Strategy | Net PnL | Fills | Quoted width | Realized spread |
| --- | ---: | ---: | ---: | ---: |
| symmetric touch | $16.80 | 1,970 | 1.99 bp | 1.55 bp |
| GLFT, gamma 1e-4 | $19.29 | 836 | 2.23 bp | 1.98 bp |
| Avellaneda-Stoikov | $9.55 | 342 | 3.66 bp | 1.69 bp |
| GLFT + imbalance | -$15.03 | 805 | 2.25 bp | 1.84 bp |

One symbol on one day is not evidence of profitability, and it is not offered as
such. Its value is as a control: the same code, the same fill model, the same
inventory bounds, applied where the quoted width is about 2 basis points instead
of 0.01 and the maker side is paid instead of charged, produces positive PnL.
That isolates the venue's spread-to-cost ratio as the operative variable rather
than any defect in the quoting logic.

## Limitations

The Avellaneda-Stoikov arm should be read narrowly. Its rolling horizon of
86,400 seconds was preregistered, and at the calibrated volatility the
`gamma sigma^2 tau / 2` term puts its half-spread near 447 USDT at session
start, roughly 41 basis points. 99.2 per cent of its quotes land beyond the
reconstructed 20-level book, where the replay declines to fill. It is reported
as preregistered rather than retuned after the fact, but its numbers describe
that parameterisation, not inventory-aware quoting in general.

The queue approximation overstates the gross loss by about an eighth, measured
on two ES days. ES is not Bybit and two days is not a distribution.

The reconstruction keeps 20 levels per side, so quotes deeper than that cannot
fill at all. This binds on one of the four arms.

Counterfactual replay ignores our own market impact. A resting order that would
have absorbed an aggressor changes nothing about the recorded tape, so every
fill is measured against a world in which we were not there.

The book and trade feeds are independent channels, and 38.8 per cent of the 427
million trade events print outside the best bid or ask of the last preceding
book snapshot. Some of that is genuine sweeping through multiple levels and some
is the book update lagging the trade that caused it; the reconstruction does not
separate the two. It bounds how precisely any fill in this study can be timed
against the state of the book, and it is published per day alongside the gap and
crossed-book counts.

Funding payments are excluded because the event stream does not carry them, the
study covers a single venue and a single contract, and the equity appendix
covers one symbol because the LOBSTER sample host no longer serves the other
four archives.

## Conclusion

The preregistered question has a clean negative answer on this venue.
Inventory-aware quoting with an order-flow skew does not earn positive net PnL
on BTCUSDT after queue position, latency, fees and adverse selection, and
markouts eat far more than the naive spread capture: about 0.3 basis points
within one second against a quoted spread near 0.01. The ordering of the
strategies is consistent with the mechanism rather than with noise, since the
rules that rest further from the touch and trade less lose less.

What the study does not say is that market making on BTCUSDT is impossible. It
says that continuous two-sided quoting at or near the touch, at a retail fee
tier, with a 100 millisecond requote floor and 50 millisecond latency, is a
losing configuration by a wide margin, and it quantifies each term in that
margin. A maker who quotes selectively, holds a materially better fee tier, or
operates inside the exchange's colocation latency is outside what was measured
here.

The more transferable output is the measurement apparatus: a fill model whose
assumptions are stated and whose error against order-level truth is quantified,
a second engine that agrees on sign and scale, and a preregistration that fixed
the 31 trials before any of them ran.
