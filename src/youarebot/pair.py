"""Совместная модель пары: один объект — диалог, три исхода.

Исходы: 0 — ботов нет, 1 — бот участник 0, 2 — бот участник 1. Ограничение «в диалоге
не больше одного бота» встроено в постановку, а не навешено пост-обработкой. Снаружи
модель выглядит как обычный бинарный классификатор участника: на вход строки участников,
на выходе P(is_bot) для каждой строки.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from youarebot.models import KEY_COL, build_pipeline


def to_dialogs(X: pd.DataFrame, swap: bool = False) -> pd.DataFrame:
    """Строки участников -> строка диалога: признаки участника 0 с суффиксом _a, 1 — _b.

    swap=True меняет участников местами (для аугментации): a — участник 1, b — участник 0.
    """
    cols = [c for c in X.columns if c != KEY_COL and not c.startswith("partner_")]
    first, second = (1, 0) if swap else (0, 1)
    a = X[X["participant_index"] == first].set_index(KEY_COL)[cols].add_suffix("_a")
    b = X[X["participant_index"] == second].set_index(KEY_COL)[cols].add_suffix("_b")
    return a.join(b, how="inner")


def dialog_labels(X: pd.DataFrame, y: np.ndarray, swap: bool = False) -> pd.Series:
    bots = pd.DataFrame({"d": X[KEY_COL].values, "p": X["participant_index"].values, "y": y})
    bots = bots[bots.y == 1].drop_duplicates("d").set_index("d")["p"]
    first = 1 if swap else 0
    # 0 — ботов нет; 1 — бот на месте a; 2 — бот на месте b
    return bots.map(lambda p: 1 if p == first else 2)


class PairClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.seed = seed

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        swaps = [False, True] if self.cfg.get("swap_augment", False) else [False]
        frames, labels = [], []
        for swap in swaps:
            d = to_dialogs(X, swap)
            frames.append(d)
            labels.append(dialog_labels(X, y, swap).reindex(d.index).fillna(0).astype(int))
        D = pd.concat(frames)
        inner_cfg = {
            "seed": self.seed,
            "features": self.cfg["features"],
            "model": self.cfg["model"],
        }
        self.pipe_ = build_pipeline(inner_cfg, list(D.columns))
        self.pipe_.fit(D, pd.concat(labels).to_numpy())
        self.classes_ = np.array([0, 1])
        self.prior_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        D = to_dialogs(X)
        proba = pd.DataFrame(
            self.pipe_.predict_proba(D), index=D.index, columns=self.pipe_.classes_
        )
        p_a = proba.get(1, pd.Series(0.0, index=D.index))
        p_b = proba.get(2, pd.Series(0.0, index=D.index))
        out = np.full(len(X), self.prior_)
        dids, idx = X[KEY_COL].to_numpy(), X["participant_index"].to_numpy()
        for i, (d, p) in enumerate(zip(dids, idx)):
            # участник без пары в выборке получает априорную долю ботов
            if d in proba.index:
                out[i] = p_a[d] if p == 0 else p_b[d]
        return np.column_stack([1 - out, out])
