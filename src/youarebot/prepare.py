"""Этап подготовки данных: raw json/csv -> parquet с признаками."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from youarebot.data import data_hash, load_split
from youarebot.features import build_frame
from youarebot.utils import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ID_COLS = ["ID", "dialog_id", "participant_index"]


def prepare(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR, llm: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"data_md5": data_hash(raw_dir), "llm_features": llm}
    splits = {split: load_split(raw_dir, split) for split in ("train", "test")}
    if llm:
        from youarebot import llm_features

        cache = out_dir / "llm_cache.parquet"
        # torch в отдельном процессе, иначе конфликт libomp с lightgbm (см. utils.py)
        subprocess.run(
            [sys.executable, "-m", "youarebot.llm_features", str(raw_dir), str(cache)], check=True
        )
        scores = pd.read_parquet(cache)
        meta["llm_observer"] = llm_features.OBSERVER
        meta["llm_performer"] = llm_features.PERFORMER
    for split, df in splits.items():
        # признаки собеседника считаем всегда, а использовать ли их, решает конфиг обучения
        frame = build_frame(df, with_partner=True)
        if llm:
            own = llm_features.participant_features(df["own_texts"], scores)
            partner = llm_features.participant_features(df["partner_texts"], scores)
            frame = pd.concat([frame, own, partner.add_prefix("partner_")], axis=1)
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
