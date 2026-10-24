"""Обучение одной модели по конфигу с логированием в MLflow."""

import argparse
from pathlib import Path

import mlflow

from youarebot.train import EXPERIMENT, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="путь к yaml-конфигу из configs/")
    args = parser.parse_args()
    mlflow.set_experiment(EXPERIMENT)
    train(args.config)
