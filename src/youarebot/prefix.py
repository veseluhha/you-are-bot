"""Онлайн-постановка: как рано можно распознать бота.

Сервис должен отвечать по ходу диалога, а не после его конца. Поэтому оцениваем модель на
префиксах диалога из первых k реплик и сравниваем два варианта обучения:
- full: обучение на полных диалогах (как в основных экспериментах);
- prefix: обучение на полных диалогах + их префиксах (аугментация данных).
"""

import json
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from youarebot.data import data_hash, dialogs_to_participants, load_dialogs
from youarebot.features import build_frame
from youarebot.models import build_pipeline, pair_constraint
from youarebot.prepare import PROJECT_ROOT, RAW_DIR
from youarebot.utils import EPS, flatten, git_commit, git_dirty, load_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL = 10_000  # «префикс» длиннее любого диалога = полный диалог


def participants_at(dialogs: dict, labels: pd.DataFrame, k: int) -> pd.DataFrame:
    """Таблица участников по первым k репликам каждого диалога."""
    cut = {d: msgs[:k] for d, msgs in dialogs.items()}
    df = dialogs_to_participants(cut).merge(labels, on=["dialog_id", "participant_index"])
    frame = build_frame(df, with_partner=True)
    frame["dialog_id"] = df["dialog_id"].values
    frame["is_bot"] = df["is_bot"].values
    return frame


def run(config_path: Path, eval_ks: list[int], train_ks: list[int]) -> dict:
    cfg = load_config(config_path)
    dialogs = load_dialogs(RAW_DIR / "train.json")
    labels = pd.read_csv(RAW_DIR / "ytrain.csv")
    frames = {k: participants_at(dialogs, labels, k) for k in sorted(set(eval_ks + train_ks))}
    full = frames[FULL]
    use_pair = cfg.get("postprocess", {}).get("pair_constraint", False)
    out_dir = PROJECT_ROOT / "outputs" / "prefix_online"
    out_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedGroupKFold(
        n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"]
    )
    y_full = full["is_bot"].to_numpy()
    preds = {
        (mode, k): pd.Series(np.nan, index=full.index)
        for mode in ("full", "prefix")
        for k in eval_ks
    }
    with mlflow.start_run(run_name="prefix_online", nested=mlflow.active_run() is not None):
        mlflow.log_params(flatten(cfg))
        mlflow.log_params({"prefix.eval_ks": str(eval_ks), "prefix.train_ks": str(train_ks)})
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "git_dirty": str(git_dirty()),
                "data_md5": data_hash(RAW_DIR),
                "config_file": config_path.name,
            }
        )
        for tr, va in cv.split(full, y_full, full["dialog_id"]):
            tr_d = set(full["dialog_id"].iloc[tr])
            for mode in ("full", "prefix"):
                ks = [FULL] if mode == "full" else train_ks
                # у копий одного диалога разной длины разные ключи: модель пары собирает
                # диалог по dialog_id, и префиксы не должны склеиваться друг с другом
                train = pd.concat(
                    frames[k][frames[k].dialog_id.isin(tr_d)].assign(
                        dialog_id=lambda f, k=k: f["dialog_id"] + f"#{k}"
                    )
                    for k in ks
                )
                X = train.drop(columns="is_bot")
                pipe = build_pipeline(cfg, list(X.columns))
                pipe.fit(X, train["is_bot"].to_numpy())
                for k in eval_ks:
                    val = frames[k][~frames[k].dialog_id.isin(tr_d)]
                    p = pipe.predict_proba(val.drop(columns="is_bot"))[:, 1]
                    if use_pair:
                        p = pair_constraint(p, val["dialog_id"])
                    preds[(mode, k)].loc[val.index] = p

        rows = []
        for (mode, k), p in preds.items():
            p = np.clip(p.to_numpy(), EPS, 1 - EPS)
            ll, auc = log_loss(y_full, p), roc_auc_score(y_full, p)
            name = "all" if k == FULL else str(k)
            mlflow.log_metric(f"{mode}_logloss_k{name}", ll)
            mlflow.log_metric(f"{mode}_roc_auc_k{name}", auc)
            rows.append({"train": mode, "k": name, "logloss": ll, "roc_auc": auc})
        table = pd.DataFrame(rows)
        table.to_csv(out_dir / "prefix_curve.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        for mode, label in (
            ("full", "обучение на полных диалогах"),
            ("prefix", "обучение + префиксы"),
        ):
            t = table[table.train == mode]
            ax.plot(range(len(t)), t["logloss"], "o-", label=label)
        ax.set_xticks(range(len(eval_ks)), [("все" if k == FULL else str(k)) for k in eval_ks])
        ax.set_xlabel("видно первых реплик диалога")
        ax.set_ylabel("logloss (OOF)")
        ax.set_title("Качество по мере развития диалога")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "prefix_curve.png", dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(out_dir / "prefix_curve.csv"), "outputs")
        mlflow.log_artifact(str(out_dir / "prefix_curve.png"), "outputs")

    print(table.to_string(index=False))
    return json.loads(table.to_json(orient="records"))
