"""Онлайн-постановка: качество модели по первым k репликам диалога."""

import argparse
from pathlib import Path

import mlflow

from youarebot.prefix import FULL, run
from youarebot.utils import EXPERIMENT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/11_pair_lgbm_swap.yaml"))
    parser.add_argument("--eval-ks", type=int, nargs="+", default=[2, 3, 4, 6, 8, FULL])
    parser.add_argument("--train-ks", type=int, nargs="+", default=[2, 4, 6, FULL])
    args = parser.parse_args()
    mlflow.set_experiment(EXPERIMENT)
    run(args.config, args.eval_ks, args.train_ks)
