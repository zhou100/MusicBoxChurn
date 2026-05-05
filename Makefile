.PHONY: install install-dev validate-data test lint format clean train train-lr train-rf train-gbm train-mlp mlflow-ui

PYTHON ?= python
DATA ?= Processed_data/df_model_final.csv
CONFIG ?= configs/train_lr.yaml

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

validate-data:
	$(PYTHON) -m musicbox_churn.data.load_data --path $(DATA)

test:
	$(PYTHON) -m pytest

train:
	$(PYTHON) -m musicbox_churn.training.train --config $(CONFIG)

train-lr:
	$(PYTHON) -m musicbox_churn.training.train --config configs/train_lr.yaml

train-rf:
	$(PYTHON) -m musicbox_churn.training.train --config configs/train_rf.yaml

train-gbm:
	$(PYTHON) -m musicbox_churn.training.train --config configs/train_gbm.yaml

train-mlp:
	$(PYTHON) -m musicbox_churn.training.train --config configs/train_mlp.yaml

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri file:./mlruns

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
