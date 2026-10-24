#!/usr/bin/env bash
# Полный эксперимент одной командой: окружение из uv.lock -> данные -> все модели -> MLflow.
set -euo pipefail
cd "$(dirname "$0")"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
export MLFLOW_DISABLE_AGENT_HINT=1

for f in train.json ytrain.csv test.json ytest.csv; do
    [[ -f "data/raw/$f" ]] || { echo "нет data/raw/$f — см. README" >&2; exit 1; }
done

uv sync --frozen
uv run python run_experiment.py "$@"
echo "MLflow UI: uv run mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI"
