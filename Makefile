PYTHON ?= python3
CONFIG ?= configs/p2_base.yaml
REPLAY_CONFIG ?= configs/p2_queue_replay.yaml

.PHONY: install test lint download-lobster simulate replay research figures clean

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	MPLBACKEND=Agg $(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts

download-lobster:
	$(PYTHON) scripts/download_lobster_samples.py

simulate:
	$(PYTHON) -m p2.simulator --config "$(CONFIG)"

replay:
	$(PYTHON) -m p2.backtest --config "$(REPLAY_CONFIG)"

research:
	$(PYTHON) scripts/run_cross_symbol_sweep.py

figures:
	$(PYTHON) scripts/render_readme_figures.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage build dist
	rm -rf p2/__pycache__ src/p2/__pycache__ tests/__pycache__ scripts/__pycache__
	rm -rf src/p2_market_maker.egg-info
