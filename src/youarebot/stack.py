"""Стекинг второго уровня: логистическая регрессия на логитах OOF-предсказаний.

Вход — OOF и сабмиты уже обученных конфигов (outputs/<run_name>/) и, по желанию,
предсказания дообученного трансформера. Второй уровень линейный и с одним коэффициентом
на модель: в отличие от передачи OOF признаком в бустинг, он не переобучается на шумный
признак (это видно по конфигу 14).
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from youarebot.data import data_hash
from youarebot.prepare import RAW_DIR
from youarebot.train import metrics
from youarebot.utils import EPS, PROJECT_ROOT, flatten, git_commit, git_dirty, load_config


def _logit(p) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))


def _inputs(cfg: dict, out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Таблицы train (OOF + метка) и test (предсказания) с колонкой на каждую модель."""
    first = cfg["members"][0]
    train = pd.read_csv(out_root / first / "oof.csv")[["ID", "dialog_id", "is_bot"]]
    test = pd.read_csv(out_root / first / "submission.csv")[["ID"]]
    for name in cfg["members"]:
        oof = pd.read_csv(out_root / name / "oof.csv").set_index("ID")["oof"]
        sub = pd.read_csv(out_root / name / "submission.csv").set_index("ID")["is_bot"]
        train[name], test[name] = train["ID"].map(oof), test["ID"].map(sub)
    if cfg.get("transformer", False):
        from youarebot.transformer import PREDS_PATH

        bert = pd.read_parquet(PREDS_PATH).set_index("ID")["bert_p"]
        train["bert_p"], test["bert_p"] = train["ID"].map(bert), test["ID"].map(bert)
    return train, test


def run(config_path: Path, out_root: Path | None = None) -> dict:
    cfg = load_config(config_path)
    out_root = out_root or PROJECT_ROOT / "outputs"
    out_dir = out_root / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = _inputs(cfg, out_root)
    cols = [c for c in train.columns if c not in ("ID", "dialog_id", "is_bot")]
    X, X_test = _logit(train[cols]), _logit(test[cols])
    y = train["is_bot"].to_numpy()

    with mlflow.start_run(run_name=cfg["run_name"], nested=mlflow.active_run() is not None):
        mlflow.log_params(flatten(cfg))
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "git_dirty": str(git_dirty()),
                "data_md5": data_hash(RAW_DIR),
                "model_name": "stack_logreg",
                "config_file": config_path.name,
            }
        )
        cv = StratifiedGroupKFold(
            n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"]
        )
        oof = np.zeros(len(y))
        for tr, va in cv.split(X, y, train["dialog_id"]):
            model = LogisticRegression(C=cfg["C"]).fit(X[tr], y[tr])
            oof[va] = model.predict_proba(X[va])[:, 1]
        result = metrics(y, oof, "cv_")
        mlflow.log_metrics(result)

        model = LogisticRegression(C=cfg["C"]).fit(X, y)
        mlflow.log_params({f"coef.{c}": round(float(w), 4) for c, w in zip(cols, model.coef_[0])})
        p_test = np.clip(model.predict_proba(X_test)[:, 1], EPS, 1 - EPS)
        sub_path = out_dir / "submission.csv"
        pd.DataFrame({"ID": test["ID"], "is_bot": p_test}).to_csv(sub_path, index=False)
        train.assign(oof=oof)[["ID", "dialog_id", "is_bot", "oof"]].to_csv(
            out_dir / "oof.csv", index=False
        )
        mlflow.log_artifact(str(sub_path), "outputs")
        mlflow.log_artifact(str(out_dir / "oof.csv"), "outputs")
        run_id = mlflow.active_run().info.run_id

    print(f"{cfg['run_name']}: " + ", ".join(f"{k}={v:.4f}" for k, v in result.items()))
    return {"run_name": cfg["run_name"], "run_id": run_id, **result, "submission": str(sub_path)}
