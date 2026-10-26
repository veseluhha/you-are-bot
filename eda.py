"""EDA: графики и сводные таблицы для презентации -> reports/."""

import json

import matplotlib
import numpy as np
import pandas as pd

from youarebot.data import load_dialogs
from youarebot.prepare import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, prepare

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = PROJECT_ROOT / "reports"
FIG = REPORTS / "figures"
COLORS = {0: "#4C72B0", 1: "#DD8452"}
LABELS = {0: "человек", 1: "бот"}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def hist_by_class(df: pd.DataFrame, col: str, title: str, name: str, bins=30, log=False):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    values = np.log1p(df[col]) if log else df[col]
    edges = np.histogram_bin_edges(values, bins=bins)
    for cls in (0, 1):
        ax.hist(
            values[df.is_bot == cls],
            bins=edges,
            alpha=0.6,
            density=True,
            color=COLORS[cls],
            label=LABELS[cls],
        )
    ax.set_title(title)
    ax.set_xlabel(f"log1p({col})" if log else col)
    ax.legend()
    save(fig, name)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if not (PROCESSED_DIR / "train.parquet").exists():
        prepare()
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    summary = {}

    # 1. баланс классов и структура пар
    per_dialog = train.groupby("dialog_id").is_bot.sum().value_counts().sort_index()
    by_index = train.groupby("participant_index").is_bot.mean()
    by_first = train.groupby("is_first").is_bot.mean()
    summary.update(
        n_dialogs_train=int(train.dialog_id.nunique()),
        n_dialogs_test=int(test.dialog_id.nunique()),
        n_participants_train=len(train),
        n_participants_test=len(test),
        bot_share=float(train.is_bot.mean()),
        bots_per_dialog={int(k): int(v) for k, v in per_dialog.items()},
        bot_share_by_index={int(k): float(v) for k, v in by_index.items()},
        bot_share_by_first_speaker={int(k): float(v) for k, v in by_first.items()},
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    train.is_bot.map(LABELS).value_counts().plot.bar(ax=axes[0], color=[COLORS[0], COLORS[1]])
    axes[0].set_title("Участники: человек / бот")
    per_dialog.plot.bar(ax=axes[1], color="#55A868")
    axes[1].set_title("Ботов в диалоге")
    axes[1].set_xlabel("")
    by_index.plot.bar(ax=axes[2], color="#8172B2")
    axes[2].set_title("Доля ботов по participant_index")
    axes[2].set_xlabel("")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    save(fig, "01_balance.png")

    # 2. train vs test: длины диалогов
    fig, ax = plt.subplots(figsize=(6, 3.5))
    edges = np.arange(0, 40, 2)
    for df, label in ((train, "train"), (test, "test")):
        lens = df.drop_duplicates("dialog_id").dialog_len.clip(upper=38)
        ax.hist(lens, bins=edges, alpha=0.6, density=True, label=label)
    ax.set_title("Длина диалога в репликах (обрезано на 38)")
    ax.legend()
    save(fig, "02_dialog_len_train_test.png")

    # 3-6. различия человек / бот
    hist_by_class(train, "len_mean", "Средняя длина реплики", "03_msg_len.png", log=True)
    hist_by_class(
        train, "echo_sim_mean", "Сходство с предыдущей репликой собеседника", "04_echo_sim.png"
    )
    hist_by_class(train, "cyr_share", "Доля реплик с кириллицей", "05_cyrillic.png", bins=10)
    hist_by_class(train, "self_sim_mean", "Сходство своих соседних реплик", "06_self_sim.png")

    # 7. средние значения признаков по классам: какие отличаются сильнее всего
    num = train.drop(columns=["ID", "dialog_id", "own_text", "partner_text"])
    num = num.loc[:, ~num.columns.str.startswith("partner_")]
    means = num.groupby("is_bot").mean().T
    std = num.drop(columns="is_bot").std()
    effect = ((means[1] - means[0]) / std).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(6, 6))
    effect.plot.barh(ax=ax, color=np.where(effect > 0, COLORS[1], COLORS[0]))
    ax.set_title("Разница средних бот − человек (в std)")
    save(fig, "07_feature_effect.png")
    means.assign(effect_std=effect).to_csv(REPORTS / "feature_means_by_class.csv")

    # качество данных
    dialogs = load_dialogs(RAW_DIR / "train.json")
    msgs = [m for d in dialogs.values() for m in d]
    summary.update(
        n_messages_train=len(msgs),
        empty_messages=sum(not m["text"].strip() for m in msgs),
        very_long_messages_gt_2000=sum(len(m["text"]) > 2000 for m in msgs),
        dialog_len_median_train=float(train.drop_duplicates("dialog_id").dialog_len.median()),
        dialog_len_median_test=float(test.drop_duplicates("dialog_id").dialog_len.median()),
        echo_contains_share_bot=float(train[train.is_bot == 1].echo_contains_share.mean()),
        echo_contains_share_human=float(train[train.is_bot == 0].echo_contains_share.mean()),
    )
    (REPORTS / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
