"""Этап подготовки данных: raw json/csv -> parquet с признаками."""

import json
from pathlib import Path

from youarebot.data import data_hash, load_split
from youarebot.features import build_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ID_COLS = ["ID", "dialog_id", "participant_index"]


def prepare(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"data_md5": data_hash(raw_dir)}
    for split in ("train", "test"):
        df = load_split(raw_dir, split)
        # признаки собеседника считаем всегда, а использовать ли их — решает конфиг обучения
        frame = build_frame(df, with_partner=True)
        keep = ID_COLS + (["is_bot"] if "is_bot" in df else [])
        for col in keep:
            frame[col] = df[col].values
        frame.to_parquet(out_dir / f"{split}.parquet", index=False)
        meta[f"n_{split}"] = len(frame)
        if "is_bot" in df:
            meta["train_bot_share"] = float(df["is_bot"].mean())
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"prepared: {meta}")
    return meta
