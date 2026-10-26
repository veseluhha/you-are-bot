"""Fine-tune rubert-tiny2 на диалоге целиком с точки зрения участника.

Вход: весь диалог, где каждая реплика размечена «Я:» (реплики участника) или
«Собеседник:», так модель видит и стиль, и взаимодействие (эхо, ответы
невпопад). Обучение по тем же 5 фолдам, что и у табличных моделей: OOF-предсказания
идут вторым уровнем в стекинг, предсказание на test: среднее моделей фолдов.
"""

import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from youarebot.data import data_hash, load_split
from youarebot.prepare import PROCESSED_DIR, RAW_DIR
from youarebot.utils import (
    EPS,
    EXPERIMENT,
    PARENT_RUN_ENV,
    flatten,
    git_commit,
    git_dirty,
    load_config,
    torch_device,
)

PREDS_PATH = PROCESSED_DIR / "transformer.parquet"


def dialog_view(turns: list[tuple], p: int) -> str:
    return "\n".join(f"{'Я' if s == p else 'Собеседник'}: {t.strip()}" for _, s, t in turns)


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(enc: dict, idx: np.ndarray, bs: int):
    for i in range(0, len(idx), bs):
        b = idx[i : i + bs]
        yield {k: v[b] for k, v in enc.items()}, b


def _fit_predict(cfg: dict, enc: dict, y: np.ndarray, tr: np.ndarray, pred_sets: list, device):
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    p = cfg["params"]
    _seed_all(cfg["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    steps = p["epochs"] * int(np.ceil(len(tr) / p["batch_size"]))
    sched = get_linear_schedule_with_warmup(opt, int(p["warmup"] * steps), steps)
    labels = torch.tensor(y)
    rng = np.random.default_rng(cfg["seed"])
    for _ in range(p["epochs"]):
        model.train()
        for batch, b in _batches(enc, rng.permutation(tr), p["batch_size"]):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch, labels=labels[b].to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    outs = []
    with torch.no_grad():
        for e, idx in pred_sets:
            res = np.zeros(len(idx))
            for batch, b in _batches(e, np.arange(len(idx)), 64):
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                res[b] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
            outs.append(res)
    return outs


def run(config_path: Path) -> dict:
    from transformers import AutoTokenizer

    cfg = load_config(config_path)
    device = torch_device()
    train = load_split(RAW_DIR, "train")
    test = load_split(RAW_DIR, "test")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])

    def encode(df: pd.DataFrame) -> dict:
        texts = [dialog_view(t, p) for t, p in zip(df["turns"], df["participant_index"])]
        return dict(
            tok(
                texts,
                truncation=True,
                max_length=cfg["params"]["max_length"],
                padding="longest",
                return_tensors="pt",
            )
        )

    enc_tr, enc_te = encode(train), encode(test)
    y = train["is_bot"].to_numpy()
    cv = StratifiedGroupKFold(
        n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"]
    )
    oof, test_pred = np.zeros(len(y)), np.zeros(len(test))

    mlflow.set_experiment(EXPERIMENT)
    # из run_experiment этап идёт отдельным процессом и вешается на родительский run
    with mlflow.start_run(run_name=cfg["run_name"], parent_run_id=os.environ.get(PARENT_RUN_ENV)):
        mlflow.log_params(flatten(cfg))
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "git_dirty": str(git_dirty()),
                "data_md5": data_hash(RAW_DIR),
                "model_name": cfg["model_name"],
                "config_file": config_path.name,
                "device": str(device),
            }
        )
        mlflow.log_artifact(str(config_path), "config")
        start = time.time()
        for fold, (tr, va) in enumerate(cv.split(train, y, train["dialog_id"])):
            enc_va = {k: v[va] for k, v in enc_tr.items()}
            p_va, p_te = _fit_predict(
                cfg, enc_tr, y, tr, [(enc_va, va), (enc_te, np.arange(len(test)))], device
            )
            oof[va] = p_va
            test_pred += p_te / cfg["cv"]["n_splits"]
            mlflow.log_metric(
                "fold_logloss", log_loss(y[va], np.clip(p_va, EPS, 1 - EPS)), step=fold
            )
        metrics = {
            "cv_logloss": log_loss(y, np.clip(oof, EPS, 1 - EPS)),
            "cv_roc_auc": roc_auc_score(y, oof),
            "train_minutes": (time.time() - start) / 60,
        }
        mlflow.log_metrics(metrics)
        preds = pd.concat(
            [
                pd.DataFrame({"ID": train["ID"], "bert_p": oof}),
                pd.DataFrame({"ID": test["ID"], "bert_p": test_pred}),
            ]
        )
        preds.to_parquet(PREDS_PATH, index=False)
        mlflow.log_artifact(str(PREDS_PATH), "outputs")
    print(f"{cfg['run_name']}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics
