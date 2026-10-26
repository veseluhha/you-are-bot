"""Target encoding реплик: как часто автор такой же реплики в трейне оказывался ботом.

Боты повторяют шаблоны («привет. как сам?» в трейне пишут только боты), и почти половина
реплик теста дословно встречается в трейне. Для каждой реплики участника берём сглаженную
долю ботов среди авторов этой реплики и агрегируем по участнику: среднее, максимум, минимум,
доля реплик, которые встречались. То же для собеседника.

Обёртка обучает признак внутри фолда: для своих обучающих строк он считается по внутренним
фолдам (иначе строка видит собственную метку), для валидации и теста по всему трейну фолда.
"""

import collections
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedGroupKFold

from youarebot.models import KEY_COL, build_pipeline

SEP = "\x1e"  # разделитель реплик в колонке own_msg_text
TE_COLS = ["te_mean", "te_max", "te_min", "te_seen"]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def template(text: str) -> str:
    """Грубее, чем normalize: без пунктуации, цифры -> 0, «saman saman saman» -> «saman *»."""
    t = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", re.sub(r"\d+", "0", text.lower()))).strip()
    words = t.split()
    return f"{words[0]} *" if len(words) > 1 and len(set(words)) == 1 else t


def split_messages(col: pd.Series, norm=normalize) -> list[list[str]]:
    return [[norm(t) for t in s.split(SEP)] if s else [] for s in col]


def fit_counts(msgs: list[list[str]], y: np.ndarray) -> tuple[dict, dict]:
    bots, total = collections.defaultdict(float), collections.defaultdict(float)
    for texts, label in zip(msgs, y):
        # одна и та же реплика участника считается один раз
        for t in set(texts):
            bots[t] += label
            total[t] += 1
    return bots, total


def encode(msgs: list[list[str]], counts: tuple[dict, dict], prior: float, smoothing: float):
    bots, total = counts
    out = np.full((len(msgs), len(TE_COLS)), prior)
    for i, texts in enumerate(msgs):
        rates = [
            (bots[t] + smoothing * prior) / (total[t] + smoothing) for t in texts if t in total
        ]
        if rates:
            out[i, :3] = np.mean(rates), max(rates), min(rates)
        out[i, 3] = len(rates) / max(len(texts), 1)
    return out


def knn_encode(
    fit_msgs: list[list[str]],
    y: np.ndarray,
    msgs: list[list[str]],
    prior: float,
    smoothing: float,
    k: int = 10,
    min_sim: float = 0.5,
) -> np.ndarray:
    """Нечёткий вариант: доля ботов среди авторов k похожих реплик трейна (косинус по
    символьным n-граммам не ниже min_sim). Ловит и вариации шаблонов, и повторы.
    Колонки: среднее и максимум по репликам участника, доля реплик с почти точным совпадением.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    pairs = [(t, label) for texts, label in zip(fit_msgs, y) for t in set(texts)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    labels = np.array([label for _, label in pairs])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(
        vec.fit_transform([t for t, _ in pairs])
    )
    out = np.zeros((len(msgs), 3))
    out[:, :2] = prior
    for i, texts in enumerate(msgs):
        if not texts:
            continue
        dist, idx = nn.kneighbors(vec.transform(texts))
        sim = 1 - dist
        w = np.where(sim >= min_sim, sim, 0.0)
        rates = ((w * labels[idx]).sum(1) + smoothing * prior) / (w.sum(1) + smoothing)
        out[i] = rates.mean(), rates.max(), (sim[:, 0] >= 0.9).mean()
    return out


def variant_columns(variants: list[str]) -> list[str]:
    cols = []
    for v in variants:
        if v == "exact":
            cols += TE_COLS
        elif v == "template":
            cols += [c.replace("te_", "te_tpl_") for c in TE_COLS]
        elif v == "knn":
            cols += ["knn_mean", "knn_max", "knn_near"]
        else:
            raise ValueError(f"неизвестный вариант target encoding: {v}")
    return cols


def encode_variants(variants, fit: dict, y: np.ndarray, apply: dict, prior, smoothing):
    """fit/apply: реплики в двух нормализациях: {"exact": [...], "template": [...]}."""
    blocks = []
    for v in variants:
        if v == "knn":
            blocks.append(knn_encode(fit["exact"], y, apply["exact"], prior, smoothing))
        else:
            blocks.append(encode(apply[v], fit_counts(fit[v], y), prior, smoothing))
    return np.column_stack(blocks)


class MessageTEClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, cfg: dict, features: dict | None = None, seed: int = 42):
        self.cfg = cfg
        self.features = features
        self.seed = seed

    def _messages(self, X: pd.DataFrame) -> dict:
        return {
            "exact": split_messages(X["own_msg_text"]),
            "template": split_messages(X["own_msg_text"], template),
        }

    def _augment(self, X: pd.DataFrame, te: np.ndarray) -> pd.DataFrame:
        own = pd.DataFrame(te, columns=self.columns_, index=X.index)
        # признаки собеседника берём из строки второго участника того же диалога
        key = pd.MultiIndex.from_arrays([X[KEY_COL], X["participant_index"]])
        partner_key = pd.MultiIndex.from_arrays([X[KEY_COL], 1 - X["participant_index"]])
        partner = own.set_axis(key).reindex(partner_key).fillna(self.prior_)
        partner = partner.set_axis(X.index).add_prefix("partner_")
        return pd.concat([X, own, partner], axis=1)

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        msgs = self._messages(X)
        smoothing = self.cfg.get("smoothing", 2.0)
        variants = self.cfg.get("variants", ["exact"])
        self.prior_ = float(y.mean())
        self.columns_ = variant_columns(variants)
        te = np.zeros((len(X), len(self.columns_)))
        inner = StratifiedGroupKFold(
            n_splits=self.cfg.get("n_splits", 5), shuffle=True, random_state=self.seed
        )

        def take(idx):
            return {k: [m[i] for i in idx] for k, m in msgs.items()}

        for tr, va in inner.split(X, y, X[KEY_COL]):
            te[va] = encode_variants(variants, take(tr), y[tr], take(va), self.prior_, smoothing)
        # для валидации и теста словари строятся заново по всему трейну фолда
        self.fit_msgs_, self.fit_y_ = msgs, y

        Xa = self._augment(X, te)
        inner_cfg = {"seed": self.seed, "features": self.features or {}, "model": self.cfg["inner"]}
        self.pipe_ = build_pipeline(inner_cfg, list(Xa.columns))
        self.pipe_.fit(Xa, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        te = encode_variants(
            self.cfg.get("variants", ["exact"]),
            self.fit_msgs_,
            self.fit_y_,
            self._messages(X),
            self.prior_,
            self.cfg.get("smoothing", 2.0),
        )
        return self.pipe_.predict_proba(self._augment(X, te))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
