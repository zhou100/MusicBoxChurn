.PHONY: install install-dev validate-data test lint format clean train train-lr train-rf train-gbm train-mlp mlflow-ui batch-score score-report monitoring-report docker-build docker-batch-score report-figures

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

# Usage: make batch-score RUN_DIR=artifacts/lr_... INPUT=Processed_data/df_model_final.csv
RUN_DIR ?= artifacts/lr_latest
INPUT ?= Processed_data/df_model_final.csv
OUTPUT_DIR ?= output
batch-score:
	$(PYTHON) -m musicbox_churn.inference.batch_score --run-dir $(RUN_DIR) --input $(INPUT) --output-dir $(OUTPUT_DIR)

score-report:
	$(PYTHON) -m musicbox_churn.monitoring.score_report --run-dir $(RUN_DIR)

monitoring-report:
	$(PYTHON) -m musicbox_churn.monitoring.drift_report --data $(DATA) --out $(OUTPUT_DIR)/monitoring_report.md

# Regenerate figures embedded in REPORT.md from the latest run of each model.
report-figures:
	$(PYTHON) scripts/generate_report_figures.py

# Container build + smoke run mirroring the k8s CronJob layout.
IMAGE ?= musicbox-churn:dev
docker-build:
	docker build -t $(IMAGE) .

# Mounts an existing run dir + the input CSV; writes to ./output.
# Usage: make docker-batch-score RUN_DIR=$$(pwd)/artifacts/rf_<ts> INPUT=$$(pwd)/Processed_data/df_model_final.csv
docker-batch-score:
	docker run --rm \
		-v $(RUN_DIR):/app/artifacts/run:ro \
		-v $$(dirname $(INPUT)):/app/input:ro \
		-v $$(pwd)/output:/app/output \
		$(IMAGE) \
		python -m musicbox_churn.inference.batch_score \
			--run-dir /app/artifacts/run \
			--input /app/input/$$(basename $(INPUT)) \
			--output-dir /app/output

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
