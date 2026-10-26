"""Признаки детекции машинного текста через перплексию LLM (идея Binoculars).

Для каждой реплики считаем на паре моделей с общим токенизатором:
- log_ppl: средний NLL токенов под базовой моделью (observer): LLM-текст «предсказуем»;
- x_ppl: кросс-энтропия между распределением instruct-модели (performer) и observer;
- binoculars = log_ppl / x_ppl, нормировка убирает зависимость от «сложности» темы,
  у машинного текста отношение заметно ниже (Hans et al., 2024).

Скоры кэшируются по md5 текста в data/processed/llm_cache.parquet, повторный прогон
пересчитывает только новые реплики.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from youarebot.utils import torch_device

OBSERVER = "Qwen/Qwen2.5-0.5B"
PERFORMER = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 128
BATCH = 8
SCORES = ["log_ppl", "x_ppl", "binoculars"]


def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class Scorer:
    """Пара observer/performer, загружается один раз на весь прогон."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch_device()
        # fp16 на GPU/MPS: две модели по 0.5B и логиты на словарь 150k токенов
        # во float32 не помещаются в память ноутбука вместе с остальным
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        self.tok = AutoTokenizer.from_pretrained(OBSERVER)
        self.observer = AutoModelForCausalLM.from_pretrained(OBSERVER, dtype=dtype)
        self.performer = AutoModelForCausalLM.from_pretrained(PERFORMER, dtype=dtype)
        self.observer.to(self.device).eval()
        self.performer.to(self.device).eval()

    def score(self, texts: list[str]) -> pd.DataFrame:
        torch, tok = self.torch, self.tok
        # у Qwen нет BOS: ставим <|endoftext|> в начало, чтобы оценивался и первый токен реплики
        enc = [
            [tok.eos_token_id] + tok(t, add_special_tokens=False)["input_ids"][: MAX_TOKENS - 1]
            for t in texts
        ]
        order = np.argsort([len(e) for e in enc])  # реплики похожей длины, меньше паддинга
        out = np.full((len(texts), len(SCORES)), np.nan)
        with torch.no_grad():
            for start in range(0, len(order), BATCH):
                idx = order[start : start + BATCH]
                width = max(len(enc[i]) for i in idx)
                ids = torch.full((len(idx), width), tok.pad_token_id)
                mask = torch.zeros((len(idx), width), dtype=torch.long)
                for row, i in enumerate(idx):
                    ids[row, : len(enc[i])] = torch.tensor(enc[i])
                    mask[row, : len(enc[i])] = 1
                ids, mask = ids.to(self.device), mask.to(self.device)
                lo = self.observer(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
                lp = self.performer(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
                target, valid = ids[:, 1:], mask[:, 1:].float()
                logp_obs = torch.log_softmax(lo, dim=-1)
                del lo
                nll = -logp_obs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                xent = -(torch.softmax(lp, dim=-1) * logp_obs).sum(-1)
                del lp, logp_obs
                n = valid.sum(1).clamp(min=1)
                log_ppl = (nll * valid).sum(1) / n
                x_ppl = (xent * valid).sum(1) / n
                scores = torch.stack([log_ppl, x_ppl, log_ppl / x_ppl], dim=1)
                out[idx] = scores.cpu().numpy()
        return pd.DataFrame(out, columns=SCORES)


def message_scores(texts: list[str], cache_path: Path, chunk: int = 512) -> pd.DataFrame:
    """Скоры для всех уникальных реплик с кэшем; индекс: md5 текста.

    Кэш дописывается после каждого куска, так что прерванный расчёт продолжается с места.
    """
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=SCORES)
    uniq = {_key(t): t for t in texts}
    missing = [k for k in uniq if k not in cache.index]
    if not missing:
        return cache
    print(f"llm: считаю перплексию для {len(missing)} реплик", flush=True)
    scorer = Scorer()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(missing), chunk):
        keys = missing[start : start + chunk]
        fresh = scorer.score([uniq[k] for k in keys])
        fresh.index = keys
        cache = pd.concat([cache, fresh]) if len(cache) else fresh
        cache.to_parquet(cache_path)
        print(f"llm: {min(start + chunk, len(missing))}/{len(missing)}", flush=True)
    return cache


def participant_features(texts_per_row: pd.Series, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for texts in texts_per_row:
        s = scores.loc[[_key(t) for t in texts if t.strip()]] if len(texts) else scores.iloc[:0]
        feats = {}
        for col in SCORES:
            v = s[col].to_numpy(dtype=float)
            feats[f"llm_{col}_mean"] = np.nanmean(v) if len(v) else np.nan
            feats[f"llm_{col}_min"] = np.nanmin(v) if len(v) else np.nan
        # короткие реплики («ок», «)») почти ничего не говорят о модели, поэтому отдельно длинные
        long = [_key(t) for t in texts if len(t.strip()) >= 20]
        feats["llm_binoculars_long_mean"] = (
            float(scores.loc[long, "binoculars"].mean()) if long else np.nan
        )
        rows.append(feats)
    return pd.DataFrame(rows)


def compute_cache(raw_dir: Path, cache_path: Path) -> None:
    """Точка входа отдельного процесса: перплексия для всех реплик train и test."""
    from youarebot.data import load_dialogs

    texts = [
        m["text"]
        for split in ("train", "test")
        for msgs in load_dialogs(raw_dir / f"{split}.json").values()
        for m in msgs
    ]
    message_scores(texts, cache_path)


if __name__ == "__main__":
    import sys

    compute_cache(Path(sys.argv[1]), Path(sys.argv[2]))
