"""Подготовка данных: data/raw -> data/processed/{train,test}.parquet."""

import argparse
from pathlib import Path

from youarebot.prepare import PROCESSED_DIR, RAW_DIR, prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    prepare(args.raw_dir, args.out_dir)
