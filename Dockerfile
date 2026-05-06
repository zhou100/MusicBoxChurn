# Batch-scoring container for musicbox_churn.
#
# Includes the package + CPU runtime only — torch is installed as the
# CPU build to keep the image lean. The default CMD invokes batch_score;
# train.py also works (override CMD).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: required by lightgbm. tini: clean PID-1 signal handling.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better layer caching). Pin torch to the CPU wheel.
COPY pyproject.toml ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch>=2.2 \
 && pip install \
        "numpy>=1.26" "pandas>=2.1" "scikit-learn>=1.4" \
        "pydantic>=2.6" "pyyaml>=6.0" "lightgbm>=4.3" \
        "mlflow>=2.12" "joblib>=1.3" "pyarrow>=15.0"

COPY src/ ./src/
COPY configs/ ./configs/
RUN pip install --no-deps -e .

# Runtime conventions:
#   /app/artifacts mounts the trained run directory
#   /app/input     mounts the CSV(s) to score
#   /app/output    mounts the destination for ranked predictions
RUN mkdir -p /app/artifacts /app/input /app/output

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "musicbox_churn.inference.batch_score", \
     "--run-dir", "/app/artifacts/run", \
     "--input",   "/app/input/scoring_input.csv", \
     "--output-dir", "/app/output"]
