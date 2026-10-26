"""Загрузка сырых диалогов и разворот в таблицу «один участник = одна строка»."""

import hashlib
import json
from pathlib import Path

import pandas as pd

RAW_FILES = ["train.json", "ytrain.csv", "test.json", "ytest.csv"]


def data_hash(raw_dir: Path) -> str:
    """md5 по всем сырым файлам, чтобы в MLflow было видно, на каких данных обучались."""
    h = hashlib.md5()
    for name in RAW_FILES:
        h.update((raw_dir / name).read_bytes())
    return h.hexdigest()


def load_dialogs(path: Path) -> dict[str, list[dict]]:
    with open(path, encoding="utf-8") as f:
        dialogs = json.load(f)
    # реплики внутри диалога сортируем по номеру, в исходнике порядок не гарантирован
    return {did: sorted(msgs, key=lambda m: m["message"]) for did, msgs in dialogs.items()}


def dialogs_to_participants(dialogs: dict[str, list[dict]]) -> pd.DataFrame:
    """Для каждого участника собираем его реплики, реплики собеседника и порядок ходов."""
    rows = []
    for did, msgs in dialogs.items():
        for p in (0, 1):
            rows.append(
                {
                    "dialog_id": did,
                    "participant_index": p,
                    # (номер реплики, кто говорит, текст): полный диалог нужен для эхо-признаков
                    "turns": [(m["message"], m["participant_index"], m["text"]) for m in msgs],
                    "own_texts": [m["text"] for m in msgs if m["participant_index"] == p],
                    "partner_texts": [m["text"] for m in msgs if m["participant_index"] != p],
                }
            )
    return pd.DataFrame(rows)


def load_split(raw_dir: Path, split: str) -> pd.DataFrame:
    dialogs = load_dialogs(raw_dir / f"{split}.json")
    df = dialogs_to_participants(dialogs)
    labels = pd.read_csv(raw_dir / f"y{split}.csv")
    # inner-join: в выборке остаются только участники, для которых есть строка в y*.csv
    df = df.merge(labels, on=["dialog_id", "participant_index"], how="inner")
    if "ID" not in df:
        df["ID"] = df["dialog_id"] + "_" + df["participant_index"].astype(str)
    return df.reset_index(drop=True)
