# P2 — Avellaneda–Stoikov Market Making with Queue-Aware Replay

[![CI](https://github.com/pdwi2020/p2_market_maker/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/pdwi2020/p2_market_maker/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository implements the Avellaneda–Stoikov (2008) market-making
policy from its HJB derivation through synthetic simulation and
queue-aware LOBSTER replay. The replay path models post-only placement,
FIFO queue priority, quote latency, maker rebates, hard inventory limits,
and forward markouts.

## What is implemented

- Closed-form reservation price and optimal spread with a documented HJB
  derivation.
- Deterministic NumPy simulation and an optional Torch accelerator path.
- Queue-reactive fill logic with cancellation and partial-fill handling.
- A Glosten–Milgrom Bayesian adverse-selection intensity layer.
- Top-of-book LOBSTER replay with post-only quote handling and latency.
- Shared cash, inventory, fee, and mark-to-market accounting.
- Baseline comparisons and a five-symbol public-sample research sweep.
- A direct replication of the central 2008 parameter experiment.

## Validated AAPL replay

The queue-aware configuration was validated on the public AAPL sample for
2012-06-21 at level 10. Raw data are not tracked in Git.

| Metric | Value |
| --- | ---: |
| Terminal PnL | $7.305 |
| Filled shares | 1,106 |
| Quoted width | $0.08022 |
| Realized spread per filled share | $0.07731 |
| Realized-spread / quoted-width ratio | 96.38% |
| Realized-spread PnL | $42.255 |
| Inventory mark-to-market | −$37.715 |
| Maker rebates | $2.765 |
| Maximum absolute inventory | 10 shares |
| Marketable quotes at placement | 0 |

The accounting components reconcile exactly:
`42.255 - 37.715 + 2.765 = 7.305`.

## Repository map

- `src/p2/hjb_solver.py` — reservation-price and spread formulas.
- `src/p2/execution.py` — shared synthetic execution and accounting.
- `src/p2/queue.py` — queue depth, FIFO, cancellation, and calibration.
- `src/p2/glosten_milgrom.py` — Bayesian adverse-selection layer.
- `src/p2/lobster_replay.py` — queue-aware LOBSTER replay engine.
- `src/p2/backtest.py` — replay runner and PnL attribution.
- `src/p2/simulator.py` — synthetic simulation and 2008 replication.
- `configs/` — deterministic simulation and replay configurations.
- `docs/hjb_derivation.md` — derivation and sign conventions.
- `tests/` — offline unit, regression, and accounting checks.

## Install and test

Python 3.11 and 3.12 are supported.

```bash
git clone https://github.com/pdwi2020/p2_market_maker.git
cd p2_market_maker
python3 -m venv .venv
source .venv/bin/activate
make install
make test
```

The default sweep runs on NumPy and does not require Torch. Optional
acceleration packages can be installed separately:

```bash
python -m pip install -e '.[gpu]'
python -m pip install -e '.[fast]'
```

## Reproducible commands

| Command | Purpose |
| --- | --- |
| `make install` | Install the package and development checks. |
| `make test` | Run the complete offline suite. |
| `make lint` | Run static checks over source, scripts, and tests. |
| `make simulate` | Run the deterministic synthetic baseline. |
| `make download-lobster` | Download all five public level-10 samples and print SHA-256 hashes. |
| `make replay` | Replay the queue-aware AAPL configuration. |
| `make research` | Run the cross-symbol baseline comparison. |
| `make figures` | Regenerate figures under the ignored `results/` tree. |
| `make clean` | Remove local build and test caches. |

`make download-lobster` obtains AAPL, AMZN, GOOG, INTC, and MSFT for
2012-06-21 directly from the official sample host. To download a subset:

```bash
python scripts/download_lobster_samples.py AAPL MSFT
```

## Data and cache paths

No external data are required for installation or the test suite. Runtime
paths are controlled through three environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `P2_DATA_DIR` | `<repo>/data` | Raw and downloaded market data. |
| `P2_CACHE_DIR` | `<repo>/.cache` | Reusable local cache files. |
| `P2_LAKE_DIR` | unset | Optional external lake; features that need it must skip when unset. |

The five replay samples come from the
[official LOBSTER sample page](https://data.lobsterdata.com/info/DataSamples.php).
Each archive contains matching message and order-book CSV files. Downloaded
data, caches, build metadata, and ordinary results are ignored. Only small,
explicitly selected derived outputs may be committed under
`results/published/`.

## Reproducibility notes

- Checked-in configurations carry fixed seeds.
- CI installs the package on Python 3.11 and 3.12 and runs without market
  data or an external lake.
- The sample downloader prints SHA-256 for each archive and extracted file.
- Relative paths resolve from the repository root, not the caller's working
  directory.

## Limitations

- The validated replay covers one public AAPL session; it is not evidence of
  cross-day or live-trading performance.
- Replay is queue-aware but remains a simplified single-venue model.
- Quote latency is configurable rather than empirically calibrated.
- The queue correction and Glosten–Milgrom layer are modular additions, not a
  jointly solved coupled HJB.
- Optional accelerator results can differ slightly across devices because of
  floating-point and random-number implementations.

## References

- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit
  order book. *Quantitative Finance*, 8(3).
- Ho, T. & Stoll, H. R. (1981). Optimal dealer pricing under transactions and
  return uncertainty. *Journal of Financial Economics*, 9(1).
- Glosten, L. R. & Milgrom, P. R. (1985). Bid, ask and transaction prices in
  a specialist market with heterogeneously informed traders. *Journal of
  Financial Economics*, 14(1).
- Huang, W., Lehalle, C.-A. & Rosenbaum, M. (2015). Simulating and analyzing
  order book data: the queue-reactive model.

## License

MIT. See [LICENSE](LICENSE).
