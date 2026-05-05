.PHONY: install install-dev validate-data test lint format clean

PYTHON ?= python
DATA ?= Processed_data/df_model_final.csv

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

validate-data:
	$(PYTHON) -m musicbox_churn.data.load_data --path $(DATA)

test:
	$(PYTHON) -m pytest

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
