"""End-to-end эксперимент: подготовка данных и обучение всех конфигов из configs/.

Все запуски попадают в MLflow как дочерние к одному родительскому run. Лучший по
cv_logloss сабмит копируется в outputs/submission.csv.
"""

import argparse
import shutil
from pathlib import Path

import mlflow
import pandas as pd

from youarebot.prepare import PROJECT_ROOT, prepare
from youarebot.train import EXPERIMENT, git_commit, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="*", help="по умолчанию все configs/*.yaml")
    args = parser.parse_args()
    configs = args.configs or sorted((PROJECT_ROOT / "configs").glob("*.yaml"))

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="pipeline") as parent:
        meta = prepare()
        mlflow.set_tags({"git_commit": git_commit(), "data_md5": meta["data_md5"]})
        mlflow.log_params({"n_configs": len(configs)})
        results = [train(path) for path in configs]

        board = pd.DataFrame(results).sort_values("cv_logloss").reset_index(drop=True)
        board_path = PROJECT_ROOT / "outputs" / "leaderboard.csv"
        board.to_csv(board_path, index=False)
        mlflow.log_artifact(str(board_path))
        best = board.iloc[0]
        shutil.copy(best["submission"], PROJECT_ROOT / "outputs" / "submission.csv")
        mlflow.set_tags({"best_run": best["run_name"], "best_run_id": best["run_id"]})
        mlflow.log_metric("best_cv_logloss", best["cv_logloss"])

    cols = ["run_name", "cv_logloss", "cv_roc_auc", "cv_accuracy", "cv_brier"]
    print(board[cols].to_string(index=False))
    print(f"best: {best['run_name']} -> outputs/submission.csv (parent run {parent.info.run_id})")
