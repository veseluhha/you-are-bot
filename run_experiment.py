"""End-to-end эксперимент одним запуском.

1. Подготовка данных: ручные признаки + перплексия LLM (кэшируется).
2. Fine-tune rubert-tiny2 по фолдам -> OOF-предсказания для стекинга (отдельный процесс).
3. Обучение всех конфигов из configs/*.yaml.
4. Стекинг второго уровня по configs/stack/*.yaml.
5. Онлайн-постановка: качество по первым k репликам диалога.

Все запуски попадают в MLflow как дочерние к одному родительскому run `pipeline`. Лучший
по cv_logloss сабмит копируется в outputs/submission.csv.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

from youarebot import prefix, stack
from youarebot.prepare import prepare
from youarebot.train import train
from youarebot.utils import EXPERIMENT, PARENT_RUN_ENV, PROJECT_ROOT, git_commit, load_config

CONFIGS = PROJECT_ROOT / "configs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="*", help="по умолчанию все configs/*.yaml")
    parser.add_argument("--no-llm", action="store_true", help="без признаков перплексии LLM")
    parser.add_argument("--no-transformer", action="store_true", help="без fine-tune rubert")
    parser.add_argument("--no-prefix", action="store_true", help="без онлайн-эксперимента")
    args = parser.parse_args()
    configs = args.configs or sorted(CONFIGS.glob("*.yaml"))
    if args.no_llm:
        configs = [c for c in configs if "llm" not in c.stem]
    if args.no_transformer:
        configs = [c for c in configs if "bert" not in c.stem]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="pipeline") as parent:
        meta = prepare(llm=not args.no_llm)
        mlflow.set_tags({"git_commit": git_commit(), "data_md5": meta["data_md5"]})
        mlflow.log_params(
            {
                "n_configs": len(configs),
                "llm": not args.no_llm,
                "transformer": not args.no_transformer,
                "prefix": not args.no_prefix,
            }
        )
        if not args.no_transformer:
            # torch — отдельным процессом (конфликт libomp с lightgbm, см. utils.py),
            # его run в MLflow вешается на этот же родительский run
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "train_transformer.py")],
                env={**os.environ, PARENT_RUN_ENV: parent.info.run_id},
                cwd=PROJECT_ROOT,
                check=True,
            )
        results = [train(path) for path in configs]
        trained = {r["run_name"] for r in results}
        for path in sorted((CONFIGS / "stack").glob("*.yaml")):
            cfg = load_config(path)
            # стекинг только поверх моделей, обученных в этом же прогоне
            if set(cfg["members"]) <= trained and not (cfg["transformer"] and args.no_transformer):
                results.append(stack.run(path))

        board = pd.DataFrame(results).sort_values("cv_logloss").reset_index(drop=True)
        board_path = PROJECT_ROOT / "outputs" / "leaderboard.csv"
        board.to_csv(board_path, index=False)
        mlflow.log_artifact(str(board_path))
        best = board.iloc[0]
        shutil.copy(best["submission"], PROJECT_ROOT / "outputs" / "submission.csv")
        mlflow.set_tags({"best_run": best["run_name"], "best_run_id": best["run_id"]})
        mlflow.log_metric("best_cv_logloss", best["cv_logloss"])

        if not args.no_prefix:
            prefix.run(
                CONFIGS / "11_pair_lgbm_swap.yaml",
                eval_ks=[2, 3, 4, 6, 8, prefix.FULL],
                train_ks=[2, 4, 6, prefix.FULL],
            )

    cols = ["run_name", "cv_logloss", "cv_roc_auc", "cv_accuracy", "cv_brier"]
    print(board[cols].to_string(index=False))
    print(f"best: {best['run_name']} -> outputs/submission.csv (parent run {parent.info.run_id})")
