VENV := /Volumes/Crucial X9/alpha_engine/.venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

CONFIG ?= /Volumes/Crucial X9/projects/p2_market_maker/configs/p2_config.yaml

install:
	"$(PIP)" install -e .[dev]

simulate:
	"$(PYTHON)" -m p2.simulator --config "$(CONFIG)"

sweep:
	"$(PYTHON)" -m p2.cuda_sweep --config "$(CONFIG)"

sweep-cross-symbol:
	"$(PYTHON)" scripts/run_cross_symbol_sweep.py

calibrate:
	"$(PYTHON)" -m p2.calibration --config "$(CONFIG)"

backtest:
	"$(PYTHON)" -m p2.backtest --config "$(CONFIG)"

test:
	"$(PYTEST)" tests/ -v --cov=src/p2 --cov-report=term-missing

notebook:
	"$(PYTHON)" -m jupyter notebook

clean:
	rm -rf __pycache__ .pytest_cache .coverage
