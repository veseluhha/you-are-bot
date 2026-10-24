"""Общие утилиты без тяжёлых зависимостей.

Модуль не импортирует ни lightgbm, ни torch: этапы на torch (перплексия LLM, fine-tune)
запускаются в отдельных процессах, потому что libomp из torch и из lightgbm/sklearn
в одном процессе на macOS приводят к segfault.
"""

import os
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-4
EXPERIMENT = "you-are-bot"
# id родительского run `pipeline` для этапов, которые идут отдельным процессом
PARENT_RUN_ENV = "YOUAREBOT_PARENT_RUN_ID"


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for i, item in enumerate(v):
                out.update(flatten(item, f"{key}.{i}."))
        else:
            out[key] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "."], cwd=PROJECT_ROOT, text=True
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
