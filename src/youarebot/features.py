"""Ручные признаки участника диалога.

Признаки считаются по одной строке и не обучаются, поэтому их можно считать один раз
до кросс-валидации, утечки между фолдами нет.
"""

import re

import numpy as np
import pandas as pd

CYR = re.compile(r"[а-яё]", re.IGNORECASE)
LAT = re.compile(r"[a-z]", re.IGNORECASE)
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")
BOT_WORDS = re.compile(r"\b(бот\w*|bot\w*|нейросет\w*|ии|ai|gpt|чат)\b", re.IGNORECASE)


def _trigrams(text: str) -> set[str]:
    t = text.lower()
    return {t[i : i + 3] for i in range(max(len(t) - 2, 1))}


def _jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa, sb = _trigrams(a), _trigrams(b)
    return len(sa & sb) / len(sa | sb)


def _text_stats(texts: list[str]) -> dict[str, float]:
    joined = " ".join(texts)
    lens = np.array([len(t) for t in texts]) if texts else np.zeros(1)
    words = np.array([len(t.split()) for t in texts]) if texts else np.zeros(1)
    n_chars = max(len(joined), 1)
    n = max(len(texts), 1)
    return {
        "n_msgs": len(texts),
        "len_mean": lens.mean(),
        "len_max": lens.max(),
        "len_std": lens.std(),
        # логарифм: длины сильно скошены (есть реплики на тысячи символов)
        "len_log_mean": np.log1p(lens).mean(),
        "words_mean": words.mean(),
        "cyr_share": sum(bool(CYR.search(t)) for t in texts) / n,
        "lat_share": sum(bool(LAT.search(t)) for t in texts) / n,
        "upper_rate": sum(c.isupper() for c in joined) / n_chars,
        "digit_rate": sum(c.isdigit() for c in joined) / n_chars,
        "punct_rate": sum(c in ".,;:!?-—" for c in joined) / n_chars,
        "newline_rate": joined.count("\n") / n,
        "emoji_rate": len(EMOJI.findall(joined)) / n,
        "question_rate": joined.count("?") / n,
        "exclam_rate": joined.count("!") / n,
        "bracket_smile_rate": joined.count(")") / n,
        "starts_upper_share": sum(t[:1].isupper() for t in texts) / n,
        "ends_dot_share": sum(t.rstrip().endswith(".") for t in texts) / n,
        "empty_share": sum(not t.strip() for t in texts) / n,
        # следы форматирования: у людей чаще пробел в конце, у ботов переносы и двойные пробелы
        "trail_ws_share": sum(t[-1:].isspace() for t in texts) / n,
        "lead_ws_share": sum(t[:1].isspace() for t in texts) / n,
        "has_newline_share": sum("\n" in t for t in texts) / n,
        "double_space_share": sum("  " in t for t in texts) / n,
        "bot_words_rate": len(BOT_WORDS.findall(joined)) / n,
        # шаблонные боты повторяют одно и то же
        "unique_share": len(set(texts)) / n,
    }


def _dialog_stats(turns: list[tuple], p: int) -> dict[str, float]:
    """Признаки, которые зависят от взаимодействия: эхо, ответ на реплику, порядок."""
    echo_sim, echo_contains, self_sim, len_ratio = [], [], [], []
    last_partner, last_own = None, None
    for _, speaker, text in turns:
        if speaker == p:
            if last_partner is not None:
                echo_sim.append(_jaccard(text, last_partner))
                echo_contains.append(
                    float(len(last_partner.strip()) > 2 and last_partner.strip() in text)
                )
                len_ratio.append(np.log1p(len(text)) - np.log1p(len(last_partner)))
            if last_own is not None:
                self_sim.append(_jaccard(text, last_own))
            last_own = text
        else:
            last_partner = text

    def mean(x):
        return float(np.mean(x)) if x else 0.0

    # самая длинная серия своих сообщений подряд: участник 1 два раза подряд пишет почти
    # только если он бот
    max_run, run = 0, 0
    for _, speaker, _ in turns:
        run = run + 1 if speaker == p else 0
        max_run = max(max_run, run)

    return {
        "is_first": float(turns[0][1] == p),
        # последнее слово в диалоге почти всегда за ботом (~90% против ~25% у людей)
        "is_last": float(turns[-1][1] == p),
        "max_run": max_run,
        "dialog_len": len(turns),
        "own_turn_share": sum(t[1] == p for t in turns) / len(turns),
        "echo_sim_mean": mean(echo_sim),
        "echo_sim_max": max(echo_sim, default=0.0),
        "echo_contains_share": mean(echo_contains),
        "self_sim_mean": mean(self_sim),
        "len_ratio_mean": mean(len_ratio),
    }


def handcrafted(df: pd.DataFrame, with_partner: bool = True) -> pd.DataFrame:
    own = pd.DataFrame([_text_stats(t) for t in df["own_texts"]])
    dial = pd.DataFrame([_dialog_stats(t, p) for t, p in zip(df["turns"], df["participant_index"])])
    feats = pd.concat([own, dial], axis=1)
    feats["participant_index"] = df["participant_index"].values
    if with_partner:
        # в диалоге не больше одного бота, поэтому признаки собеседника тоже информативны
        partner = pd.DataFrame([_text_stats(t) for t in df["partner_texts"]]).add_prefix("partner_")
        feats = pd.concat([feats, partner], axis=1)
    return feats.astype(float)


def build_frame(df: pd.DataFrame, with_partner: bool = True) -> pd.DataFrame:
    """Итоговая таблица признаков: тексты для TF-IDF + числовые признаки."""
    out = handcrafted(df, with_partner=with_partner)
    out["own_text"] = [" \n ".join(t) for t in df["own_texts"]]
    out["partner_text"] = [" \n ".join(t) for t in df["partner_texts"]]
    # реплики по отдельности (разделитель \x1e) для target encoding реплик, см. message_te.py
    out["own_msg_text"] = ["\x1e".join(t) for t in df["own_texts"]]
    out["partner_msg_text"] = ["\x1e".join(t) for t in df["partner_texts"]]
    return out
