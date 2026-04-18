# Cross-symbol quoter ablation on LOBSTER 2012-06-21

## Methodology pivot

Free LOBSTER provides one shared trading day, `2012-06-21`, across the public sample symbols rather than a true multi-day panel. As a result, this Week 2 study pivots from the original five-day robustness design to a cross-sectional sweep across symbols on the same date. That change should be stated plainly in the memo: this is not evidence about day-to-day stability. Instead, it is a complementary robustness test asking whether the Avellaneda-Stoikov quoting logic and its baselines behave consistently across distinct names with different price levels, liquidity, and event intensities when calibrated from the same sample day.

## Quoters compared

The AVS-optimal quoter is the reference strategy. The ablation compares it against four baselines:

- `SymmetricMM`: fixed-width bid and ask quotes around the mid with no inventory skew.
- `ConstantSpreadMM`: fixed-width quotes using the AVS time-zero half-spread, removing dynamic adaptation.
- `InventoryLinearMM`: fixed-width quotes with a linear inventory skew that narrows the side needed to offload risk.
- `RandomQuoter`: independent random quote distances that serve as a noise-floor baseline.

## Results table

Filled by `make sweep-cross-symbol`; table lives at `results/cross_symbol_ablation.csv`.

## Next

The natural extension is a true multi-day panel using paid LOBSTER access, then a broader validation pass on richer market data such as NASDAQ ITCH or WRDS TAQ to separate symbol effects from day effects.
