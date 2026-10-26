"""Этап обучения: кросс-валидация по диалогам, финальная модель, сабмит, логи в MLflow."""

import json
from pathlib import Path

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from youarebot.models import build_pipeline, numeric_columns, pair_constraint
from youarebot.prepare import ID_COLS, PROCESSED_DIR
from youarebot.utils import (  # noqa: F401  EXPERIMENT реэкспортируется для скриптов
    EPS,
    EXPERIMENT,
    PROJECT_ROOT,
    flatten,
    git_commit,
    git_dirty,
    load_config,
    set_seed,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def metrics(y: np.ndarray, p: np.ndarray, prefix: str) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    return {
        f"{prefix}logloss": log_loss(y, p, labels=[0, 1]),
        f"{prefix}roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        f"{prefix}accuracy": accuracy_score(y, p > 0.5),
        f"{prefix}brier": brier_score_loss(y, p),
    }


def calibration_plot(y: np.ndarray, p: np.ndarray, path: Path) -> None:
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    frac = [y[idx == i].mean() if (idx == i).any() else np.nan for i in range(10)]
    conf = [p[idx == i].mean() if (idx == i).any() else np.nan for i in range(10)]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(conf, frac, "o-")
    ax.set_xlabel("предсказанная вероятность")
    ax.set_ylabel("доля ботов")
    ax.set_title("Калибровка OOF")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def feature_importance(pipe, path: Path) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    if "prep" not in pipe.named_steps:
        return None
    names = pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = model.coef_[0]
    else:
        return None
    fi = pd.DataFrame({"feature": names, "importance": imp})
    fi = fi.reindex(fi["importance"].abs().sort_values(ascending=False).index)
    fi.to_csv(path, index=False)
    return fi


def add_transformer_preds(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет OOF-предсказание трансформера для участника и его собеседника, если оно есть."""
    from youarebot.transformer import PREDS_PATH

    if not PREDS_PATH.exists():
        return df
    preds = pd.read_parquet(PREDS_PATH).set_index("ID")["bert_p"]
    partner_id = df["dialog_id"] + "_" + (1 - df["participant_index"]).astype(str)
    return df.assign(bert_p=df["ID"].map(preds).values, partner_bert_p=partner_id.map(preds).values)


def train(config_path: Path, out_root: Path | None = None) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    run_name = cfg["run_name"]
    out_dir = (out_root or PROJECT_ROOT / "outputs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = add_transformer_preds(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
    test_df = add_transformer_preds(pd.read_parquet(PROCESSED_DIR / "test.parquet"))
    meta = json.loads((PROCESSED_DIR / "meta.json").read_text())
    # dialog_id остаётся во входе модели: по нему модель пары собирает диалог
    feat_df = train_df.drop(columns=["ID", "is_bot"])
    columns = list(feat_df.columns)
    num_cols = numeric_columns(columns, cfg.get("features", {}))
    y = train_df["is_bot"].to_numpy()
    groups = train_df["dialog_id"]
    use_pair = cfg.get("postprocess", {}).get("pair_constraint", False)

    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None):
        mlflow.log_params(flatten(cfg))
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "git_dirty": str(git_dirty()),
                "data_md5": meta["data_md5"],
                "model_name": cfg["model"]["name"],
                "config_file": config_path.name,
            }
        )
        mlflow.log_params(
            {
                "data.n_train": meta["n_train"],
                "data.n_test": meta["n_test"],
                "data.n_numeric_features": len(num_cols),
            }
        )
        mlflow.log_artifact(str(config_path), "config")
        mlflow.log_artifact(str(PROCESSED_DIR / "meta.json"), "data")

        # группировка по dialog_id: оба участника диалога всегда в одном фолде,
        # иначе признаки собеседника и пост-обработка пары дают утечку
        cv = StratifiedGroupKFold(
            n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"]
        )
        oof = np.zeros(len(y))
        for fold, (tr, va) in enumerate(cv.split(feat_df, y, groups)):
            pipe = build_pipeline(cfg, columns)
            pipe.fit(feat_df.iloc[tr], y[tr])
            p = pipe.predict_proba(feat_df.iloc[va])[:, 1]
            if use_pair:
                p = pair_constraint(p, groups.iloc[va])
            oof[va] = p
            mlflow.log_metric("fold_logloss", log_loss(y[va], np.clip(p, EPS, 1 - EPS)), step=fold)

        result = metrics(y, oof, "cv_")
        mlflow.log_metrics(result)

        pipe = build_pipeline(cfg, columns)
        pipe.fit(feat_df, y)
        p_train = pipe.predict_proba(feat_df)[:, 1]
        if use_pair:
            p_train = pair_constraint(p_train, groups)
        mlflow.log_metrics(metrics(y, p_train, "train_"))

        test_X = test_df[feat_df.columns]
        p_test = pipe.predict_proba(test_X)[:, 1]
        if use_pair:
            p_test = pair_constraint(p_test, test_df["dialog_id"])
        p_test = np.clip(p_test, EPS, 1 - EPS)
        mlflow.log_metric("test_pred_mean", float(p_test.mean()))

        sub_path = out_dir / "submission.csv"
        pd.DataFrame({"ID": test_df["ID"], "is_bot": p_test}).to_csv(sub_path, index=False)
        oof_path = out_dir / "oof.csv"
        train_df[ID_COLS + ["is_bot"]].assign(oof=oof).to_csv(oof_path, index=False)
        calib_path = out_dir / "calibration.png"
        calibration_plot(y, oof, calib_path)
        for path in (sub_path, oof_path, calib_path):
            mlflow.log_artifact(str(path), "outputs")
        fi = feature_importance(pipe, out_dir / "feature_importance.csv")
        if fi is not None:
            mlflow.log_artifact(str(out_dir / "feature_importance.csv"), "outputs")

        example = feat_df.head(5)
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            # skops по умолчанию не доверяет типам lightgbm, модели свои, берём cloudpickle
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=example,
            # сервис ждёт вероятности, а не метки: pyfunc-модель отдаёт predict_proba
            pyfunc_predict_fn="predict_proba",
            signature=infer_signature(example, pipe.predict_proba(example)),
        )
        run_id = mlflow.active_run().info.run_id

    print(f"{run_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in result.items()))
    return {"run_name": run_name, "run_id": run_id, **result, "submission": str(sub_path)}
