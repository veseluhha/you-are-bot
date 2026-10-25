"""Сборка sklearn-пайплайна по конфигу эксперимента."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KEY_COL = "dialog_id"


def is_text(col: str) -> bool:
    # own_text, partner_text и их варианты own_text_a / own_text_b у модели пары
    return "_text" in col


def numeric_columns(columns: list[str], features_cfg: dict) -> list[str]:
    if not features_cfg.get("handcrafted", True):
        return []
    cols = [c for c in columns if c != KEY_COL and not is_text(c)]
    if not features_cfg.get("partner_features", True):
        cols = [c for c in cols if not c.startswith("partner_")]
    if not features_cfg.get("llm", False):
        # признаки перплексии LLM включаются явно, чтобы старые конфиги не меняли смысл
        cols = [c for c in cols if "llm_" not in c]
    if not features_cfg.get("transformer", False):
        # OOF-предсказание дообученного трансформера — признак второго уровня (стекинг)
        cols = [c for c in cols if "bert_p" not in c]
    return cols


def _tfidf(params: dict) -> TfidfVectorizer:
    params = dict(params)
    params["ngram_range"] = tuple(params.get("ngram_range", (1, 1)))
    return TfidfVectorizer(dtype=np.float32, **params)


def build_pipeline(cfg: dict, columns: list[str]) -> Pipeline:
    model_cfg = cfg["model"]
    name, params = model_cfg["name"], dict(model_cfg.get("params", {}))
    seed = cfg["seed"]

    if name == "prior":
        # бейзлайн: доля ботов в трейне, без признаков
        return Pipeline([("model", DummyClassifier(strategy="prior"))])
    if name == "blend":
        # усреднение вероятностей нескольких моделей, у каждой свой набор признаков
        members = [
            (f"m{i}_{m['model']['name']}", build_pipeline({"seed": seed, **m}, columns))
            for i, m in enumerate(model_cfg["members"])
        ]
        blend = VotingClassifier(members, voting="soft", weights=model_cfg.get("weights"))
        return Pipeline([("model", blend)])
    if name == "pair":
        from youarebot.pair import PairClassifier

        return Pipeline([("model", PairClassifier(model_cfg, seed))])
    if name == "message_te":
        from youarebot.message_te import MessageTEClassifier

        # обёртка: добавляет target encoding реплик и обучает внутреннюю модель model.inner
        return Pipeline([("model", MessageTEClassifier(model_cfg, cfg.get("features"), seed))])

    features_cfg = cfg["features"]
    num_cols = numeric_columns(columns, features_cfg)
    transformers = []
    if num_cols:
        # линейной модели нужны отмасштабированные признаки, бустингу — нет
        scaler = StandardScaler() if name == "logreg" else "passthrough"
        transformers.append(("num", scaler, num_cols))
    for col in filter(is_text, columns):
        tfidf_cfg = features_cfg.get(f"{col}_tfidf")
        if tfidf_cfg:
            transformers.append((f"{col}_tfidf", _tfidf(tfidf_cfg), col))
    if not transformers:
        raise ValueError("конфиг не задаёт ни одного признака")
    prep = ColumnTransformer(transformers, sparse_threshold=1.0)

    if name == "logreg":
        model = LogisticRegression(random_state=seed, **params)
    elif name == "lgbm":
        n_seeds = model_cfg.get("seeds", 1)
        model = LGBMClassifier(random_state=seed, verbose=-1, **params)
        if n_seeds > 1:
            # усреднение бустингов с разными сидами (subsample/colsample) — меньше разброс
            seeds = [
                (f"s{i}", LGBMClassifier(random_state=seed + i, verbose=-1, **params))
                for i in range(n_seeds)
            ]
            model = VotingClassifier(seeds, voting="soft")
    else:
        raise ValueError(f"неизвестная модель: {name}")
    return Pipeline([("prep", prep), ("model", model)])


def pair_constraint(proba: np.ndarray, dialog_ids: pd.Series) -> np.ndarray:
    """Пересчёт вероятностей с учётом того, что в диалоге не больше одного бота.

    Предсказания участников считаем независимыми и оставляем три допустимых исхода:
    ботов нет, бот — участник 0, бот — участник 1. Маргинал участника после нормировки —
    p_i * (1 - p_j) / Z. Если в диалоге один размеченный участник, вероятность не меняется.
    """
    out = proba.copy()
    df = pd.DataFrame({"d": dialog_ids.values, "p": proba})
    for idx in df.groupby("d").indices.values():
        if len(idx) != 2:
            continue
        p0, p1 = proba[idx[0]], proba[idx[1]]
        a, b, none = p0 * (1 - p1), (1 - p0) * p1, (1 - p0) * (1 - p1)
        z = a + b + none
        out[idx[0]], out[idx[1]] = a / z, b / z
    return out
